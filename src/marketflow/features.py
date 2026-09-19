from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_div(a, b):
    return a / b.replace(0, np.nan)


def add_stock_features(prices: pd.DataFrame, benchmark: pd.DataFrame) -> pd.DataFrame:
    p = prices.copy().sort_values(['ticker', 'date'])
    b = benchmark.copy().sort_values('date')
    b = b[['date', 'close']].rename(columns={'close': 'benchmark_close'})
    for w in [5, 20, 60, 120]:
        b[f'benchmark_ret_{w}'] = b['benchmark_close'].pct_change(w)
    p = p.merge(b, on='date', how='left')

    g = p.groupby('ticker', group_keys=False)
    p['ret_1'] = g['close'].pct_change()
    for w in [5, 20, 60, 120]:
        p[f'ret_{w}'] = g['close'].pct_change(w)
        p[f'rs_{w}'] = p[f'ret_{w}'] - p[f'benchmark_ret_{w}']

    for w in [20, 50, 200]:
        p[f'ma{w}'] = g['close'].transform(lambda s: s.rolling(w, min_periods=w).mean())
        p[f'ma{w}_distance'] = _safe_div(p['close'], p[f'ma{w}']) - 1

    p['ma20_slope_5'] = g['ma20'].pct_change(5, fill_method=None)
    p['ma50_slope_10'] = g['ma50'].pct_change(10, fill_method=None)
    p['volume_avg_20'] = g['volume'].transform(lambda s: s.rolling(20, min_periods=10).mean())
    p['volume_ratio_20'] = _safe_div(p['volume'], p['volume_avg_20'])
    if 'value' not in p.columns:
        p['value'] = p['close'] * p['volume']
    p['value_avg_20'] = g['value'].transform(lambda s: s.rolling(20, min_periods=10).mean())
    p['value_ratio_20'] = _safe_div(p['value'], p['value_avg_20'])
    p['turnover_proxy'] = p['value']
    p['turnover_proxy_avg_20'] = g['turnover_proxy'].transform(lambda s: s.rolling(20, min_periods=10).mean())
    p['turnover_proxy_accel_20'] = g['turnover_proxy_avg_20'].pct_change(20, fill_method=None)
    p['volatility_20'] = g['ret_1'].transform(lambda s: s.rolling(20, min_periods=15).std())
    p['positive_day_share_20'] = g['ret_1'].transform(lambda s: (s > 0).rolling(20, min_periods=15).mean())
    posvol = p['volume'].where(p['ret_1'] > 0, 0.0)
    p['_posvol'] = posvol
    p['up_volume_share_20'] = g['_posvol'].transform(lambda s: s.rolling(20, min_periods=10).sum()) / g['volume'].transform(lambda s: s.rolling(20, min_periods=10).sum())
    p['high_252'] = g['close'].transform(lambda s: s.rolling(252, min_periods=60).max())
    p['high_252_proximity'] = _safe_div(p['close'], p['high_252'])
    p['drawdown_60'] = _safe_div(p['close'], g['close'].transform(lambda s: s.rolling(60, min_periods=20).max())) - 1

    # Technical confirmation layer used only for the entry-candidate screen.
    # MACD uses standard 12/26 EMAs with a 9-period signal line.
    p['ema12'] = g['close'].transform(lambda s: s.ewm(span=12, adjust=False, min_periods=12).mean())
    p['ema26'] = g['close'].transform(lambda s: s.ewm(span=26, adjust=False, min_periods=26).mean())
    p['macd'] = p['ema12'] - p['ema26']
    p['macd_signal'] = p.groupby('ticker')['macd'].transform(
        lambda s: s.ewm(span=9, adjust=False, min_periods=9).mean()
    )
    p['macd_hist'] = p['macd'] - p['macd_signal']
    p['macd_positive'] = (p['macd'] > 0) & (p['macd_hist'] > 0)

    # Bullish MA structure plus a recent bullish cross bonus.
    p['ma_bull'] = p['ma20'] > p['ma50']
    prev_ma_bull = p.groupby('ticker')['ma_bull'].shift(1).fillna(False)
    p['ma_cross_up'] = p['ma_bull'] & (~prev_ma_bull)
    p['ma_cross_recent_10'] = p.groupby('ticker')['ma_cross_up'].transform(
        lambda s: s.astype(float).rolling(10, min_periods=1).max()
    ).fillna(0).astype(bool)

    # Bollinger squeeze: bandwidth in the bottom 20% of its own trailing
    # 120-session distribution. A breakout gets a bonus, but is not required.
    p['bb_std20'] = g['close'].transform(lambda s: s.rolling(20, min_periods=20).std())
    p['bb_upper'] = p['ma20'] + 2.0 * p['bb_std20']
    p['bb_lower'] = p['ma20'] - 2.0 * p['bb_std20']
    p['bb_bandwidth'] = _safe_div(p['bb_upper'] - p['bb_lower'], p['ma20'])
    p['bb_bandwidth_q20_120'] = p.groupby('ticker')['bb_bandwidth'].transform(
        lambda s: s.rolling(120, min_periods=60).quantile(0.20)
    )
    p['bb_squeeze'] = p['bb_bandwidth'] <= p['bb_bandwidth_q20_120']
    p['_bb_squeeze_prev'] = p.groupby('ticker')['bb_squeeze'].shift(1).fillna(False)
    p['bb_squeeze_recent_10'] = p.groupby('ticker')['_bb_squeeze_prev'].transform(
        lambda s: s.astype(float).rolling(10, min_periods=1).max()
    ).fillna(0).astype(bool)
    prev_bw = p.groupby('ticker')['bb_bandwidth'].shift(1)
    p['bb_breakout_after_squeeze'] = (
        p['bb_squeeze_recent_10']
        & (p['close'] > p['bb_upper'])
        & (p['bb_bandwidth'] > prev_bw)
    )

    return p.drop(columns=['_posvol','_bb_squeeze_prev'])


def add_sector_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    for w in [20, 60]:
        sec_med = x.groupby(['date', 'sector'])[f'ret_{w}'].transform('median')
        x[f'sector_ret_{w}'] = sec_med
        x[f'rs_sector_{w}'] = x[f'ret_{w}'] - sec_med

    x['_above_ma20'] = (x['close'] > x['ma20']).astype(float)
    x['_above_ma50'] = (x['close'] > x['ma50']).astype(float)
    x['breadth_ma20'] = x.groupby(['date', 'sector'])['_above_ma20'].transform('mean')
    x['breadth_ma50'] = x.groupby(['date', 'sector'])['_above_ma50'].transform('mean')
    x['sector_value'] = x.groupby(['date', 'sector'])['value'].transform('sum')
    sec = x[['date', 'sector', 'sector_value']].drop_duplicates().sort_values(['sector', 'date'])
    sec['sector_value_avg20'] = sec.groupby('sector')['sector_value'].transform(lambda s: s.rolling(20, min_periods=10).mean())
    sec['sector_value_ratio_20'] = sec['sector_value'] / sec['sector_value_avg20'].replace(0, np.nan)
    x = x.merge(sec[['date', 'sector', 'sector_value_ratio_20']], on=['date', 'sector'], how='left')
    x['sector_rs_20'] = x['sector_ret_20'] - x['benchmark_ret_20']
    x['sector_rs_60'] = x['sector_ret_60'] - x['benchmark_ret_60']
    return x.drop(columns=['_above_ma20', '_above_ma50'])
