from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

from marketflow.risk import simulate_position_lifecycle, stop_sensitivity_study


def test_risk_engine_cuts_loser_and_rides_winner():
    dates = pd.bdate_range('2026-01-02', periods=8)
    entries = pd.DataFrame([
        {'entry_date': dates[0], 'ticker':'LOSS', 'sector':'A', 'entry_reason':'test'},
        {'entry_date': dates[0], 'ticker':'WIN', 'sector':'B', 'entry_reason':'test'},
    ])
    rows = []
    loss = [100, 96, 94, 93, 92, 91, 90]
    win = [100, 104, 109, 110, 109, 111, 110]
    for ticker, closes in [('LOSS', loss), ('WIN', win)]:
        for i, d in enumerate(dates[1:]):
            cl = closes[i]
            low = cl - (2 if ticker == 'LOSS' and i == 1 else .5)
            rows.append({
                'date': d, 'ticker': ticker, 'open': closes[i], 'high': cl+.5,
                'low': low, 'close': cl, 'ma20': cl-1, 'ma20_slope_5': .01,
                'weekly_trend_confirm': True,
            })
    px = pd.DataFrame(rows)
    scores = px[['date','ticker']].copy()
    scores['real_strength_score'] = 80
    scores['leadership_score'] = 80
    scores['trend_score'] = 80

    out = simulate_position_lifecycle(
        px, scores, entries,
        {'hard_stop_pct':.05,'profit_arm_pct':.08,'profit_floor_pct':.02,
         'peak_trail_pct':.07,'strength_break_votes':2}
    )
    loss_row = out[out['ticker'].eq('LOSS')].iloc[0]
    win_row = out[out['ticker'].eq('WIN')].iloc[0]
    assert loss_row['status'] == 'CLOSED'
    assert str(loss_row['exit_reason']).startswith('HARD_STOP')
    assert abs(loss_row['lifecycle_return'] + .05) < 1e-9
    assert win_row['status'] == 'OPEN'
    assert win_row['action'] == 'RIDE TREND'
    assert win_row['profit_protection_armed']
    assert win_row['protective_stop'] > win_row['fill_price']


def test_strength_break_exits_next_open_after_profit_is_armed():
    dates = pd.bdate_range('2026-02-02', periods=7)
    entries = pd.DataFrame([{'entry_date':dates[0],'ticker':'AAA','sector':'A'}])
    closes = [100,105,110,109,108,107]
    rows=[]
    for i,d in enumerate(dates[1:]):
        cl=closes[i]
        rows.append({'date':d,'ticker':'AAA','open':cl,'high':cl+.5,'low':cl-.5,
                     'close':cl,'ma20':cl-1,'ma20_slope_5':.01,
                     'weekly_trend_confirm':True})
    px=pd.DataFrame(rows)
    scores=px[['date','ticker']].copy()
    scores['real_strength_score']=[80,80,80,50,50,50]
    scores['leadership_score']=[80,80,80,50,50,50]
    scores['trend_score']=80
    out=simulate_position_lifecycle(
        px,scores,entries,
        {'hard_stop_pct':.05,'profit_arm_pct':.08,'profit_floor_pct':.02,
         'peak_trail_pct':.20,'strength_break_votes':2,
         'strength_real_floor':60,'strength_leadership_floor':60}
    )
    row=out.iloc[0]
    assert row['status']=='CLOSED'
    assert str(row['exit_reason']).startswith('STRENGTH_BREAK')
    # Break is observed at a close and executed causally at the next open.
    assert pd.Timestamp(row['exit_date']) > dates[4]


def test_stop_sensitivity_reports_false_stops_and_rescued_losers():
    dates = pd.bdate_range('2026-03-02', periods=5)
    entries = pd.DataFrame([
        {'entry_date':dates[0],'ticker':'A'},
        {'entry_date':dates[0],'ticker':'B'},
    ])
    rows=[]
    paths={'A':[100,94,90,90], 'B':[100,94,105,110]}
    for ticker,vals in paths.items():
        for i,d in enumerate(dates[1:]):
            cl=vals[i]
            rows.append({'date':d,'ticker':ticker,'open':vals[i],'high':cl+1,
                         'low':cl-1,'close':cl,'ma20':cl,'ma20_slope_5':0,
                         'weekly_trend_confirm':True})
    px=pd.DataFrame(rows)
    scores=px[['date','ticker']].copy()
    scores['real_strength_score']=80
    scores['leadership_score']=80
    scores['trend_score']=80
    study=stop_sensitivity_study(px,scores,entries,stop_grid=(.05,))
    row=study.iloc[0]
    assert row['stopped_count']==2
    assert row['false_stop_winners']==1
    assert row['rescued_losers']==1
