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



def test_atr_adaptive_stop_widens_for_high_volatility_stock():
    dates = pd.bdate_range('2026-04-01', periods=5)
    entries = pd.DataFrame([
        {'entry_date': dates[0], 'ticker':'LOW'},
        {'entry_date': dates[0], 'ticker':'HIGH'},
    ])
    rows=[]
    for ticker, atr in [('LOW', .02), ('HIGH', .05)]:
        vals=[100,100,96,97,98]
        for i,d in enumerate(dates):
            low = vals[i]-.5
            if i == 2:
                low = 94.5
            rows.append({
                'date':d,'ticker':ticker,'open':vals[i],'high':vals[i]+1,
                'low':low,'close':vals[i],'atr_pct_20':atr,'atr_regime_ratio':1.0,
                'ma20':95,'ma20_slope_5':.01,'weekly_trend_confirm':True,
            })
    px=pd.DataFrame(rows)
    scores=px[['date','ticker']].copy()
    scores['real_strength_score']=80
    scores['leadership_score']=80
    scores['trend_score']=80

    out=simulate_position_lifecycle(
        px,scores,entries,
        {'adaptive_volatility':True,'stop_atr_multiple':2.0,
         'min_stop_pct':.05,'max_stop_pct':.12,
         'min_profit_arm_pct':.10,'max_profit_arm_pct':.22,'profit_arm_r':1.5,
         'trail_atr_multiple':2.5,'min_trail_pct':.08,'max_trail_pct':.18,
         'strength_break_votes':2}
    )
    low=out[out['ticker'].eq('LOW')].iloc[0]
    high=out[out['ticker'].eq('HIGH')].iloc[0]
    assert abs(low['adaptive_stop_pct']-.05) < 1e-9
    assert abs(high['adaptive_stop_pct']-.10) < 1e-9
    assert low['status']=='CLOSED'
    assert str(low['exit_reason']).startswith('VOL_ADAPTIVE_STOP')
    assert high['status']=='OPEN'
    assert abs(low['profit_arm_pct_used']-.10) < 1e-9
    assert abs(high['profit_arm_pct_used']-.15) < 1e-9



def test_structure_stop_avoids_bvh_style_false_cut_before_breakout():
    dates = pd.bdate_range('2025-12-08', periods=10)
    entry = pd.DataFrame([{
        'entry_date': dates[0], 'ticker':'BVH', 'sector':'Bảo hiểm',
        'entry_reason':'SM ngắn hạn vượt 80'
    }])
    # Signal-day context resembles BVH 08/12/2025: ATR ~2.7%, recent base low
    # around 50.4, then a shakeout toward 51 before a strong advance.
    closes = [55.6,55.0,54.3,54.6,52.0,52.7,53.4,53.4,54.0,56.0]
    lows   = [52.0,54.2,53.1,53.4,52.0,51.6,51.0,53.0,52.3,53.7]
    rows=[]
    for i,d in enumerate(dates):
        rows.append({
            'date':d,'ticker':'BVH','open':56.5 if i==1 else closes[i],
            'high':max(closes[i]+1,lows[i]+1),'low':lows[i],'close':closes[i],
            'atr_pct_20':.0270413669,'atr_regime_ratio':.91,
            'prior_low_10':50.4,'prior_low_20':50.4,
            'ma20':52.0,'ma20_slope_5':.01,'weekly_trend_confirm':True,
        })
    px=pd.DataFrame(rows)
    scores=px[['date','ticker']].copy()
    scores['real_strength_score']=80
    scores['leadership_score']=80
    scores['trend_score']=80

    atr_only=simulate_position_lifecycle(
        px,scores,entry,
        {'adaptive_volatility':True,'use_structural_stop':False,
         'stop_atr_multiple':2.0,'min_stop_pct':.05,'max_stop_pct':.12,
         'min_profit_arm_pct':.10,'max_profit_arm_pct':.22,'profit_arm_r':1.5,
         'strength_break_votes':2}
    ).iloc[0]
    structure=simulate_position_lifecycle(
        px,scores,entry,
        {'adaptive_volatility':True,'use_structural_stop':True,
         'structure_buffer_atr':.25,'stop_atr_multiple':2.0,
         'min_stop_pct':.05,'max_stop_pct':.12,
         'reference_position_stop_pct':.05,
         'min_profit_arm_pct':.10,'max_profit_arm_pct':.22,'profit_arm_r':1.5,
         'strength_break_votes':2}
    ).iloc[0]

    assert atr_only['status']=='CLOSED'
    assert structure['status']=='OPEN'
    assert structure['risk_mode']=='STRUCTURE_ATR'
    assert structure['structural_stop_pct'] > structure['atr_stop_pct']
    assert .10 < structure['adaptive_stop_pct'] < .12
    assert structure['position_size_factor_vs_5pct'] < .5
