from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

from marketflow.selection import detect_opportunity_entries, build_model_portfolio, build_entry_candidates


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


def test_entry_candidates_require_macd_and_ma_and_bonus_squeeze():
    latest = pd.DataFrame([
        {
            'ticker':'AAA','sector':'Bank','sector_score':88,'leadership_score':90,
            'short_momentum_score':92,'long_momentum_score':86,'flow_score':89,'trend_score':88,
            'macd':1.2,'macd_signal':0.8,'macd_hist':0.4,'macd_positive':True,
            'ma20':30,'ma50':28,'ma_bull':True,'ma_cross_recent_10':False,
            'bb_breakout_after_squeeze':True,'stage':'LEADER',
        },
        {
            'ticker':'BBB','sector':'Tech','sector_score':85,'leadership_score':91,
            'short_momentum_score':90,'long_momentum_score':90,'flow_score':90,'trend_score':90,
            'macd':1.0,'macd_signal':0.7,'macd_hist':0.3,'macd_positive':True,
            'ma20':50,'ma50':47,'ma_bull':True,'ma_cross_recent_10':True,
            'bb_breakout_after_squeeze':False,'stage':'LEADER',
        },
        {
            'ticker':'CCC','sector':'Oil','sector_score':95,'leadership_score':95,
            'short_momentum_score':95,'long_momentum_score':95,'flow_score':95,'trend_score':95,
            'macd':-0.2,'macd_signal':-0.3,'macd_hist':0.1,'macd_positive':False,
            'ma20':40,'ma50':38,'ma_bull':True,'ma_cross_recent_10':True,
            'bb_breakout_after_squeeze':True,'stage':'LEADER',
        },
    ])
    out = build_entry_candidates(latest, top_n=3)
    assert set(out['ticker']) == {'AAA','BBB'}
    assert 'CCC' not in out['ticker'].tolist()
    assert out.iloc[0]['ticker'] == 'AAA'
    assert out.iloc[0]['technical_setup'] == 'BB squeeze breakout'
