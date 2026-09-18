from __future__ import annotations

import numpy as np
import pandas as pd


def pct_rank(df: pd.DataFrame, col: str) -> pd.Series:
    return df.groupby('date')[col].rank(pct=True, method='average') * 100.0


def weighted_score(df: pd.DataFrame, spec: dict, reverse_negative: bool = True) -> pd.Series:
    parts = []
    total = 0.0
    for col, w in spec.items():
        if col not in df.columns:
            continue
        r = pct_rank(df, col)
        if w < 0 and reverse_negative:
            r = 100.0 - r
            w = abs(w)
        parts.append(r * w)
        total += abs(w)
    if not parts or total == 0:
        return pd.Series(np.nan, index=df.index)
    return sum(parts) / total


def build_scores(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    x = df.copy()
    w = cfg['weights']
    x['rs_score'] = weighted_score(x, w['relative_strength'])
    x['flow_score'] = weighted_score(x, w['flow'])
    x['trend_score'] = weighted_score(x, w['trend'])

    sector_cols = ['date', 'sector'] + list(w['sector'].keys())
    sec = x[sector_cols].drop_duplicates(['date', 'sector']).copy()
    sec['sector_score'] = weighted_score(sec, w['sector'])
    x = x.merge(sec[['date', 'sector', 'sector_score']], on=['date', 'sector'], how='left')

    lw = w['leadership']
    x['leadership_score'] = sum(x[k] * v for k, v in lw.items()) / sum(lw.values())
    accw = int(cfg['model'].get('acceleration_window', 5))
    x = x.sort_values(['ticker', 'date'])
    x['acceleration'] = x.groupby('ticker')['leadership_score'].diff(accw)
    return x


def classify_stage(row: pd.Series, stages: dict) -> str:
    s = row.get('leadership_score', np.nan)
    a = row.get('acceleration', np.nan)
    if pd.isna(s):
        return 'INSUFFICIENT_DATA'
    if s >= stages['mature_score'] and (pd.isna(a) or abs(a) < 5):
        return 'MATURE'
    if s >= stages['leader_score'] and (pd.isna(a) or a > stages['fading_acceleration']):
        return 'LEADER'
    if s >= stages['emerging_min_score'] and not pd.isna(a) and a >= stages['emerging_acceleration']:
        return 'EMERGING'
    if (not pd.isna(a) and a <= stages['fading_acceleration']) or s < stages['fading_score']:
        return 'FADING'
    return 'NEUTRAL'


def add_stage(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    out = df.copy()
    out['stage'] = out.apply(classify_stage, axis=1, stages=cfg['stages'])
    return out
