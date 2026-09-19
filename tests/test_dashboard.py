from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

from marketflow.dashboard import render_dashboard


def _latest():
    return pd.DataFrame({
        'date': pd.to_datetime(['2026-09-18']*6),
        'ticker': ['AAA','BBB','CCC','DDD','EEE','FFF'],
        'sector': ['Ngân hàng','Ngân hàng','Công nghệ','Điện','BĐS','BĐS'],
        'leadership_score': [92,85,78,68,55,48],
        'acceleration': [8,2,12,11,-4,-14],
        'rs_score': [90,82,80,72,50,45],
        'flow_score': [88,75,84,68,48,40],
        'trend_score': [91,80,74,65,52,42],
        'sector_score': [86,86,79,70,58,58],
        'stage': ['LEADER','MATURE','EMERGING','EMERGING','NEUTRAL','FADING'],
    })


def test_professional_dashboard_renders_core_charts(tmp_path):
    latest = _latest()
    dates = pd.date_range('2026-07-01', periods=60, freq='B')
    regime = pd.DataFrame({
        'date': dates,
        'market_score': [45+i*.4 for i in range(60)],
        'breadth_ma20': [0.4+i*.005 for i in range(60)],
        'breadth_ma50': [0.35+i*.004 for i in range(60)],
    })
    sectors = pd.DataFrame([
        {'date': d, 'sector': s, 'sector_score': 50+i*.3+j*4}
        for i,d in enumerate(dates[-40:])
        for j,s in enumerate(['Ngân hàng','Công nghệ','Điện'])
    ])
    sectors['acceleration'] = sectors.groupby('sector')['sector_score'].diff(5)

    deciles = pd.DataFrame({
        'horizon':[20]*10, 'metric':['market_alpha']*10,
        'decile':range(1,11), 'mean':[x/1000 for x in range(-5,5)],
    })
    ic = pd.DataFrame({
        'date':dates[-20:], 'horizon':[20]*20, 'metric':['market_alpha']*20,
        'ic':[(-.1+i*.01) for i in range(20)],
    })
    summary = pd.DataFrame({
        'horizon':[5,20,60],
        'top_bottom_spread':[.01,.03,.05],
        'top_decile_hit_rate':[.54,.58,.61],
        'mean_spearman_ic':[.04,.07,.09],
    })

    out = tmp_path/'dashboard.html'
    render_dashboard(
        latest,
        {'regime':'RISK_ON','market_score':71,'breadth_ma20':.68,'breadth_ma50':.61,
         'coverage_stocks':190,'coverage_reference':200},
        out,
        regime_history=regime,
        sector_history=sectors,
        backtest={'deciles':deciles,'ic_daily':ic,'summary':summary},
    )
    text = out.read_text(encoding='utf-8')
    assert 'Xu hướng regime & độ rộng' in text
    assert 'Luân chuyển ngành' in text
    assert 'Điểm dẫn dắt × tăng tốc' in text
    assert 'Alpha theo decile' in text
    assert '<svg' in text
