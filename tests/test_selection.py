from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

from marketflow.selection import detect_opportunity_entries, build_model_portfolio, build_entry_candidates, build_entry_signal_history, build_entry_watchlist


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




def test_entry_v3_requires_persistent_htf_strength_and_rejects_chase():
    dates = pd.date_range('2026-09-07', periods=10, freq='B')
    rows = []

    # AAA has persistent real strength, then resumes above MA20 on the last day.
    for i,d in enumerate(dates):
        close = 30.0 + i*.05
        ma20 = 30.1
        if i == 8:
            close = 30.05
        if i == 9:
            close = 30.25
        rows.append({
            'date':d,'ticker':'AAA','sector':'Bank','close':close,
            'sector_score':78,'leadership_score':82,'short_momentum_score':76,
            'long_momentum_score':80,'flow_score':82,'trend_score':84,
            'ret_5':.03,'ma20':ma20,'ma20_distance':close/ma20-1,'ma50':28.5,
            'macd':.45,'macd_hist':[-.08,-.07,-.06,-.05,-.04,-.03,-.02,-.01,-.005,.09][i],
            'macd_positive':i==9,'volume_ratio_20':1.15,
            'bb_squeeze_recent_10':False,'bb_breakout_after_squeeze':False,
            'prior_high_10':31.0,'medium_trend_confirm':True,'weekly_trend_confirm':True,
            'weekly_ret12':.12,'rs_60':.10,'rs_120':.16,'rs_sector_60':.05,'sector_rs_60':.06,
            'stage':'LEADER',
        })

    # BBB generated a valid pullback entry several sessions ago, but price is now
    # more than 5% above that entry and therefore must not be a fresh candidate.
    for i,d in enumerate(dates):
        close = [40,40.1,40.2,40.3,40.2,40.4,41.2,42.0,43.0,44.0][i]
        ma20 = 40.3
        rows.append({
            'date':d,'ticker':'BBB','sector':'Tech','close':close,
            'sector_score':86,'leadership_score':88,'short_momentum_score':82,
            'long_momentum_score':84,'flow_score':88,'trend_score':86,
            'ret_5':.04 if i<7 else .09,'ma20':ma20,'ma20_distance':close/ma20-1,'ma50':38.0,
            'macd':.6,'macd_hist':[-.03,-.02,-.01,.01,.02,.08,.09,.1,.11,.12][i],
            'macd_positive':i>=3,'volume_ratio_20':1.2,
            'bb_squeeze_recent_10':False,'bb_breakout_after_squeeze':False,
            'prior_high_10':45.0,'medium_trend_confirm':True,'weekly_trend_confirm':True,
            'weekly_ret12':.15,'rs_60':.12,'rs_120':.18,'rs_sector_60':.06,'sector_rs_60':.08,
            'stage':'LEADER',
        })

    hist = pd.DataFrame(rows)
    events = build_entry_signal_history(hist)
    assert set(events['ticker']) == {'AAA','BBB'}
    assert events[events['ticker'].eq('AAA')]['entry_date'].iloc[-1] == dates[-1]
    assert events[events['ticker'].eq('BBB')]['entry_date'].iloc[-1] == dates[5]

    out = build_entry_candidates(hist, top_n=3, max_age_sessions=2, max_distance_from_entry=.05)
    assert out['ticker'].tolist() == ['AAA']
    assert out.iloc[0]['entry_age_sessions'] == 0
    assert abs(out.iloc[0]['since_entry_pct']) < 1e-9



def test_entry_v4_requires_residual_strength_not_only_raw_rs():
    dates = pd.date_range('2026-09-07', periods=10, freq='B')
    rows = []
    for ticker, residual in [('GOOD', .08), ('BETA', -.02)]:
        for i,d in enumerate(dates):
            close = 30.0 + i*.02
            if i == 8:
                close = 30.02
            if i == 9:
                close = 30.25
            rows.append({
                'date':d,'ticker':ticker,'sector':'Bank','close':close,
                'sector_score':80,'leadership_score':85,'real_strength_score':82,
                'short_momentum_score':78,'long_momentum_score':82,
                'flow_score':80,'trend_score':84,'ret_5':.03,
                'ma20':30.1,'ma20_distance':close/30.1-1,'ma50':28.5,
                'macd':.4,'macd_hist':[-.08,-.07,-.06,-.05,-.04,-.03,-.02,-.01,.01,.08][i],
                'macd_positive':i==9,'volume_ratio_20':1.15,
                'bb_squeeze_recent_10':False,'bb_breakout_after_squeeze':False,
                'prior_high_10':31.0,'medium_trend_confirm':True,
                'weekly_trend_confirm':True,'weekly_ret12':.12,
                'rs_60':.10,'rs_120':.15,'rs_sector_60':.05,'sector_rs_60':.06,
                'residual_mom_60':residual,'residual_mom_120':residual,
                'rs_persistence_count':3,'path_quality_60':.20,
                'return_concentration_60':.18,'stage':'LEADER',
            })
    hist = pd.DataFrame(rows)
    events = build_entry_signal_history(
        hist,
        min_real_strength_score=70,
        min_rs_persistence=2,
        require_residual_momentum=True,
    )
    assert events['ticker'].tolist() == ['GOOD']
    assert events.iloc[0]['real_strength_med10'] >= 70
    assert events.iloc[0]['residual_mom_60'] > 0



def test_watchlist_ranks_near_entries_and_explains_failed_gates():
    dates = pd.bdate_range('2026-09-07', periods=10)
    rows = []
    for ticker, residual, weekly in [('AAA', .08, True), ('BBB', -.01, True), ('CCC', .05, False)]:
        for i,d in enumerate(dates):
            rows.append({
                'date':d,'ticker':ticker,'sector':'Bank','close':30+i*.05,
                'sector_score':80,'leadership_score':85,'real_strength_score':82,
                'short_momentum_score':78,'long_momentum_score':82,
                'flow_score':80,'trend_score':84,'ret_5':.03,
                'ma20_distance':.02,'medium_trend_confirm':True,
                'weekly_trend_confirm':weekly,'rs_60':.10,'rs_120':.15,
                'rs_sector_60':.05,'sector_rs_60':.06,
                'residual_mom_60':residual,'residual_mom_120':residual,
                'rs_persistence_count':3,'stage':'LEADER',
            })
    out = build_entry_watchlist(
        pd.DataFrame(rows), top_n=3, min_real_strength_score=70,
        min_rs_persistence=2, require_residual_momentum=True,
    )
    assert out.iloc[0]['ticker'] == 'AAA'
    assert out.iloc[0]['gate_fail_count'] == 0
    assert out[out['ticker'].eq('BBB')]['gate_failures'].iloc[0].startswith('Residual')
    assert 'Weekly trend' in out[out['ticker'].eq('CCC')]['gate_failures'].iloc[0]
