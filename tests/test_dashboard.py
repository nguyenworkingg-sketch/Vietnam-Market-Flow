from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

from marketflow.dashboard import render_dashboard, _candlestick_macd_svg


def _latest():
    return pd.DataFrame({
        'date': pd.to_datetime(['2026-09-18']*6),
        'ticker': ['AAA','BBB','CCC','DDD','EEE','FFF'],
        'sector': ['Ngân hàng','Ngân hàng','Công nghệ','Điện','BĐS','BĐS'],
        'close': [31.0,29.0,27.0,25.0,23.0,21.0],
        'leadership_score': [92,85,78,68,55,48],
        'short_momentum_score': [88,82,84,76,61,42],
        'long_momentum_score': [86,83,79,81,58,45],
        'acceleration': [8,2,12,11,-4,-14],
        'rs_score': [90,82,80,72,50,45],
        'flow_score': [88,75,84,68,48,40],
        'trend_score': [91,80,74,65,52,42],
        'sector_score': [86,86,79,70,58,58],
        'stage': ['LEADER','MATURE','EMERGING','EMERGING','NEUTRAL','FADING'],
        'macd': [1.2,1.0,.8,.6,.2,-.1],
        'macd_signal': [.8,.7,.5,.4,.2,0],
        'macd_hist': [.4,.3,.3,.2,0,-.1],
        'macd_positive': [True,True,True,True,False,False],
        'ma20': [30,28,26,24,22,20],
        'ma50': [28,27,25,23,23,21],
        'ma_bull': [True,True,True,True,False,False],
        'ma_cross_up': [False,False,False,False,False,False],
        'ma_cross_recent_10': [False,True,False,False,False,False],
        'ma20_distance': [.033,.036,.038,.042,.04,.03],
        'ret_5': [.04,.04,.05,.04,.02,-.01],
        'volume_ratio_20': [1.2,1.1,1.2,1.1,.9,.8],
        'prior_high_10': [32,31,29,27,25,23],
        'bb_squeeze_recent_10': [False,False,False,False,False,False],
        'bb_breakout_after_squeeze': [False,False,False,False,False,False],
        'medium_trend_confirm': [True,True,True,True,False,False],
        'weekly_trend_confirm': [True,True,True,True,False,False],
        'weekly_ret12': [.12,.10,.11,.09,.02,-.01],
        'rs_60': [.10,.08,.09,.07,.01,-.02],
        'rs_120': [.15,.12,.13,.10,.02,-.03],
        'rs_sector_60': [.05,.04,.05,.03,.0,-.01],
        'sector_rs_60': [.06,.06,.05,.04,.0,-.02],
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

    history_frames = []
    hist_dates = pd.bdate_range(end='2026-09-18', periods=10)
    for i,d in enumerate(hist_dates):
        h = latest.copy()
        h['date'] = d
        # Keep broad strength persistent; create one clean AAA pullback-resume
        # trigger on the final session.
        h['close'] = [30.15+i*.03, 28.5+i*.02, 26.5+i*.02, 24.5+i*.02, 23, 21]
        h['ma20'] = [30.0,28.0,26.0,24.0,22.0,20.0]
        h['ma20_distance'] = h['close']/h['ma20']-1
        h['macd_hist'] = [-.08+.005*i,.18,.16,.14,0,-.1]
        h['macd_positive'] = [False,True,True,True,False,False]
        h['short_momentum_score'] = [82,82,80,76,61,42]
        h['long_momentum_score'] = [84,83,79,81,58,45]
        if i == 8:
            h.loc[h['ticker'].eq('AAA'),'close'] = 30.05
            h.loc[h['ticker'].eq('AAA'),'ma20_distance'] = 30.05/30.0-1
            h.loc[h['ticker'].eq('AAA'),'macd_hist'] = .02
        if i == 9:
            h.loc[h['ticker'].eq('AAA'),'close'] = 30.40
            h.loc[h['ticker'].eq('AAA'),'ma20_distance'] = 30.40/30.0-1
            h.loc[h['ticker'].eq('AAA'),'macd_hist'] = .09
            h.loc[h['ticker'].eq('AAA'),'macd_positive'] = True
        history_frames.append(h)
    scored_history = pd.concat(history_frames, ignore_index=True)
    latest = scored_history[scored_history['date'].eq(pd.Timestamp('2026-09-18'))].copy()

    price_rows=[]
    for ticker in latest['ticker']:
        base=20+len(price_rows)*2
        for i,d in enumerate(dates):
            close=base+i*.08
            price_rows.append({
                'date':d,'ticker':ticker,'open':close-.1,'high':close+.3,
                'low':close-.25,'close':close,'volume':1_000_000+i*10_000,
            })
    price_history=pd.DataFrame(price_rows)

    historical_entries = pd.DataFrame([{
        'entry_date': pd.Timestamp('2026-07-15'),
        'ticker': 'AAA',
        'sector': 'Ngân hàng',
        'entry_price': 25.0,
        'entry_score': 80.0,
        'entry_reason': 'Legacy test',
        'model_version': 'v2-causal-2026-09-20',
    }])

    out = tmp_path/'dashboard.html'
    render_dashboard(
        latest,
        {'regime':'RISK_ON','market_score':71,'breadth_ma20':.68,'breadth_ma50':.61,
         'coverage_stocks':190,'coverage_reference':200},
        out,
        scored_history=scored_history,
        regime_history=regime,
        sector_history=sectors,
        backtest={'deciles':deciles,'ic_daily':ic,'summary':summary},
        price_history=price_history,
        historical_entry_events=historical_entries,
    )
    text = out.read_text(encoding='utf-8')
    assert 'Xu hướng regime & độ rộng' in text
    assert 'Luân chuyển ngành' in text
    assert 'Cơ hội mới vào Top hôm nay' in text
    assert 'Model Portfolio — 10 cổ phiếu mạnh' in text
    assert 'SM ngắn hạn' in text
    assert 'Xếp hạng ngành & cổ phiếu dẫn dắt' in text
    assert 'Bấm vào từng ngành để xem top 5 cổ phiếu' in text
    assert 'AAA' in text
    assert 'Điểm dẫn dắt × tăng tốc' in text
    assert 'Biểu đồ nến · Volume · MACD' in text
    assert 'MACD (12,26,9)' in text
    assert 'Biểu đồ kỹ thuật toàn bộ cổ phiếu' in text
    assert "id='stock-chart-select'" in text
    assert "id='stock-chart-range'" in text
    assert '12 tháng' in text
    assert 'v2-causal-2026-09-20' in text
    assert 'Alpha theo decile' in text
    assert '<svg' in text


def test_entry_marker_is_rendered_above_chart():
    dates = pd.bdate_range(end='2026-09-18', periods=40)
    rows = []
    for i,d in enumerate(dates):
        close = 20 + i*.08
        rows.append({
            'date': d, 'ticker': 'AAA',
            'open': close-.05, 'high': close+.2, 'low': close-.2,
            'close': close, 'volume': 1_000_000+i*1000,
        })
    px = pd.DataFrame(rows)
    entry = pd.DataFrame([{
        'entry_date': pd.Timestamp('2026-09-18'),
        'ticker': 'AAA',
        'entry_price': float(px.iloc[-1]['close']),
        'entry_score': 88.5,
        'entry_reason': 'Pullback resume + HTF confirm',
    }])
    svg = _candlestick_macd_svg(px, 'AAA', days=40, entry_events=entry)
    assert 'ENTRY 18/09' in svg
    assert 'Từ entry:' in svg



def test_dashboard_falls_back_to_watchlist_when_no_fresh_entry(tmp_path):
    latest = _latest()
    dates = pd.bdate_range(end='2026-09-18', periods=60)
    frames = []
    for d in pd.bdate_range(end='2026-09-18', periods=10):
        h = latest.copy()
        h['date'] = d
        h['real_strength_score'] = [85,80,78,72,55,45]
        h['residual_mom_60'] = [.12,.10,.08,.05,-.01,-.02]
        h['residual_mom_120'] = [.20,.16,.12,.08,-.02,-.03]
        h['rs_persistence_count'] = [3,3,3,3,1,0]
        h['path_quality_60'] = [.20,.18,.17,.15,.05,.02]
        frames.append(h)
    scored = pd.concat(frames, ignore_index=True)
    latest = scored[scored['date'].eq(scored['date'].max())].copy()

    price_rows = []
    for j,ticker in enumerate(latest['ticker']):
        for i,d in enumerate(dates):
            close = 20+j*2+i*.05
            price_rows.append({
                'date':d,'ticker':ticker,'open':close-.1,'high':close+.2,
                'low':close-.2,'close':close,'volume':1_000_000+i*1000,
            })
    out = tmp_path/'dashboard-watchlist.html'
    render_dashboard(
        latest,
        {'regime':'NEUTRAL','market_score':50,'breadth_ma20':.5,'breadth_ma50':.5},
        out,
        scored_history=scored,
        price_history=pd.DataFrame(price_rows),
        opportunity_cfg={
            'entry_min_real_strength_score': 99,
            'entry_min_rs_persistence': 3,
            'entry_require_residual_momentum': True,
            'entry_require_medium_trend': True,
            'entry_require_weekly_trend': True,
        },
    )
    text = out.read_text(encoding='utf-8')
    assert '0 FRESH ENTRY' in text
    assert 'Top 3 Near Entry / Watchlist' in text
    assert 'WATCHLIST — NOT ENTRY' in text
    assert 'model không ép đủ Top 3' in text
