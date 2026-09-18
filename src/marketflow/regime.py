from __future__ import annotations

import numpy as np
import pandas as pd


def market_regime(df: pd.DataFrame, benchmark: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    rows = []
    for dt, d in x.groupby('date'):
        valid = d.dropna(subset=['ma20', 'ma50'])
        if valid.empty:
            continue
        breadth20 = (valid['close'] > valid['ma20']).mean()
        breadth50 = (valid['close'] > valid['ma50']).mean()
        median20 = valid['ret_20'].median()
        liq = valid['value_ratio_20'].median()
        b = benchmark[benchmark['date'] <= dt].sort_values('date').tail(60)
        idx20 = np.nan
        idx50 = np.nan
        if len(b) >= 50:
            px = float(b.iloc[-1]['close'])
            idx20 = px / b['close'].tail(20).mean() - 1
            idx50 = px / b['close'].tail(50).mean() - 1
        score = 50
        score += 20 * (breadth20 - 0.5)
        score += 15 * (breadth50 - 0.5)
        score += 120 * np.nan_to_num(median20, nan=0.0)
        score += 60 * np.nan_to_num(idx20, nan=0.0)
        score += 40 * np.nan_to_num(idx50, nan=0.0)
        score += 5 * (np.nan_to_num(liq, nan=1.0) - 1.0)
        score = float(np.clip(score, 0, 100))
        label = 'RISK_ON' if score >= 65 else ('DEFENSIVE' if score < 40 else 'NEUTRAL')
        rows.append({'date': dt, 'market_score': score, 'breadth_ma20': breadth20,
                     'breadth_ma50': breadth50, 'median_ret20': median20,
                     'liquidity_ratio': liq, 'regime': label})
    return pd.DataFrame(rows)
