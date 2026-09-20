from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

from marketflow.selection import detect_opportunity_entries, build_model_portfolio, build_entry_candidates, build_entry_signal_history


def test_detects_new_short_and_long_entries():
    d1 = pd.Timestamp('2026-09-17')
    d2 = pd.Timestamp('2026-09-18')
    hist = pd.DataFrame([
        {'date':d1,'ticker':'AAA','sector':'Bank','short_momentum_score':75,'long_momentum_score':82,'leadership_score':70,'acceleration':2,'flow_score':70,'trend_score':80,'sector_score':70,'stage':'NEUTRAL'},
        {'date':d2,'ticker':'AAA','sector':'Bank','short_momentum_score':84,'long_momentum_score':85,'leadership_score':82,'acceleration':8,'flow_score':85,'trend_score':84,'sector_score':75,'stage':'EMERGING'},
        {'date':d1,'ticker':'BBB','sector':'Tech','short_momentum_score':85,'long_momentum_score':72,'leadership_score':78,'acceleration':3,'flow_score':82,'trend_score':72,'sector_score':76,'stage':'LEADER'},
        {'date':d2,'ticker':'BBB','sector':'Tech','short_momentum_score':88,'long_momentum_score':83,'leadership_score':86,'acceleration':6,'flow_score':88,'trend_score':80,'sector_score':80,'stage':'LEADER'},
        {'date':d1,'ticker':'CCC','sector':'Weak','short_momentum_score':70,'long_momentum_score':70,'leadership_score':60,'acceleration':1,'flow_score':65,'trend_score':65,'sector_score':45,'stage':'NEUTRAL'},
        {'date':d2,'ticker':'CCC','sector':'Weak','short_momentum_score':90,'long_momentum_score':90,'leadership_score':88,'acceleration':12,'flow_score':90,'trend_score':90,'sector_score':45,'stage':'LEADER'},
    ])
    short, long = detect_opportunity_entries(hist, 80, 80, 50)
    assert short['ticker'].tolist() == ['AAA']
    assert long['ticker'].tolist() == ['BBB']


def test_model_portfolio_is_sector_first_and_capped():
    rows = []
    for sidx, sector in enumerate(['S1','S2','S3','S4','S5','S6']):
        for j in range(3):
            rows.append({
                'date':pd.Timestamp('2026-09-18'),
                'ticker':f'{sector}{j}',
                'sector':sector,
                'sector_score':95-sidx*7,
                'leadership_score':90-j*4-sidx,
                'short_momentum_score':88-j*2,
                'long_momentum_score':86-j*2,
                'flow_score':84-j*2,
                'acceleration':8-j,
                'stage':'LEADER',
            })
    latest = pd.DataFrame(rows)
    port = build_model_portfolio(latest, size=10, sector_cap=2)
    assert len(port) == 10
    assert port.groupby('sector').size().max() <= 2
    assert set(['portfolio_rank','sector_rank','portfolio_score','weight']).issubset(port.columns)
    assert abs(port['weight'].sum() - 100) < 1e-9



def test_entry_signals_are_fresh_and_reject_late_chase():
    dates = pd.date_range('2026-09-10', periods=5, freq='B')
    rows = []
    # AAA gets a fresh short-momentum/MACD trigger on the last session.
    for i,d in enumerate(dates):
        rows.append({
            'date':d,'ticker':'AAA','sector':'Bank','close':30+i*.3,
            'sector_score':80,'leadership_score':78+i,
            'short_momentum_score':[72,74,76,79,84][i],
            'long_momentum_score':82,'flow_score':84,'trend_score':82,
            'ret_5':.08,'ma20_distance':.09,'macd':.5,
            'macd_hist':[-.2,-.1,-.05,-.02,.2][i],
            'macd_positive':i==4,'ma_bull':True,'ma_cross_up':False,
            'bb_breakout_after_squeeze':False,'ma20':28,'ma50':27,'stage':'LEADER',
        })
    # BBB had a valid MA cross four sessions ago but has since run +25% and is stale.
    for i,d in enumerate(dates):
        rows.append({
            'date':d,'ticker':'BBB','sector':'Tech','close':[40,41,44,47,50][i],
            'sector_score':88,'leadership_score':90,
            'short_momentum_score':88,'long_momentum_score':86,'flow_score':90,'trend_score':88,
            'ret_5':[.04,.05,.10,.16,.25][i],
            'ma20_distance':[.07,.08,.11,.17,.24][i],
            'macd':1.0,'macd_hist':.3,'macd_positive':True,'ma_bull':True,
            'ma_cross_up':i==0,'bb_breakout_after_squeeze':False,
            'ma20':39,'ma50':38,'stage':'LEADER',
        })
    hist = pd.DataFrame(rows)
    events = build_entry_signal_history(hist)
    assert set(events['ticker']) == {'AAA','BBB'}
    assert events[events['ticker'].eq('BBB')]['entry_date'].iloc[-1] == dates[0]

    out = build_entry_candidates(hist, top_n=3, max_age_sessions=3, max_distance_from_entry=.08)
    assert out['ticker'].tolist() == ['AAA']
    assert out.iloc[0]['entry_age_sessions'] == 0
    assert abs(out.iloc[0]['since_entry_pct']) < 1e-9
