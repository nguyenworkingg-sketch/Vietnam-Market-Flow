from __future__ import annotations

import pandas as pd


def add_forward_returns(df: pd.DataFrame, horizons=(5, 20, 60)) -> pd.DataFrame:
    x = df.copy().sort_values(['ticker', 'date'])
    g = x.groupby('ticker', group_keys=False)
    for h in horizons:
        x[f'fwd_{h}'] = g['close'].shift(-h) / x['close'] - 1
    return x


def decile_study(df: pd.DataFrame, horizons=(5, 20, 60)) -> pd.DataFrame:
    x = add_forward_returns(df, horizons)
    x['score_decile'] = x.groupby('date')['leadership_score'].transform(
        lambda s: pd.qcut(s.rank(method='first'), 10, labels=False, duplicates='drop') + 1
    )
    cols = ['score_decile'] + [f'fwd_{h}' for h in horizons]
    out = x[cols].groupby('score_decile').agg(['mean', 'median', 'count'])
    return out
