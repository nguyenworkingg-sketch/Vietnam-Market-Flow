from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def fail(message: str) -> None:
    raise SystemExit(f'DATA QUALITY FAILED: {message}')


def main() -> None:
    scores_path = ROOT/'outputs'/'scores_latest.csv'
    regime_path = ROOT/'outputs'/'market_regime.csv'
    if not scores_path.exists() or not regime_path.exists():
        fail('missing scores_latest.csv or market_regime.csv')

    scores = pd.read_csv(scores_path)
    regime = pd.read_csv(regime_path)
    if scores.empty:
        fail('latest score snapshot is empty')
    if regime.empty:
        fail('market regime history is empty')

    if scores.duplicated(['date', 'ticker']).any():
        fail('duplicate ticker/date rows in latest snapshot')

    score_cols = ['rs_score','flow_score','trend_score','sector_score','leadership_score']
    for col in score_cols:
        if col not in scores:
            fail(f'missing {col}')
        bad = scores[col].dropna().loc[lambda s: (s < -1e-9) | (s > 100 + 1e-9)]
        if len(bad):
            fail(f'{col} has values outside [0,100]')

    latest_date = pd.to_datetime(scores['date']).max().normalize()
    r = regime[pd.to_datetime(regime['date']).dt.normalize().eq(latest_date)]
    if r.empty:
        fail('regime does not contain the same authoritative as-of date as scores')
    r = r.iloc[-1]

    coverage = int(r.get('coverage_stocks', len(scores)))
    reference = int(r.get('coverage_reference', coverage))
    if reference > 0 and coverage / reference < 0.70:
        fail(f'cross-section coverage only {coverage}/{reference}')

    if len(scores) < 50:
        # A normal production run should never be this thin. Explicit smoke
        # tests do not invoke this validator.
        fail(f'production snapshot contains only {len(scores)} stocks')

    if scores['ticker'].nunique() != len(scores):
        fail('ticker duplication detected')

    unknown = scores['sector'].fillna('').str.contains('Chưa phân ngành', case=False).mean()
    if unknown > 0.05:
        fail(f'{unknown:.1%} of scored names have unknown sector')

    if scores['sector'].nunique() < 5:
        fail('sector breadth is implausibly narrow')

    q = scores['leadership_score'].quantile([0.1, 0.5, 0.9])
    if q.loc[0.9] - q.loc[0.1] < 15:
        fail('leadership score distribution is too compressed to be informative')

    print(
        'DATA QUALITY PASSED | '
        f'asof={latest_date.date()} stocks={len(scores)} '
        f'coverage={coverage}/{reference} sectors={scores["sector"].nunique()} '
        f'leadership_p10/p50/p90={q.loc[0.1]:.1f}/{q.loc[0.5]:.1f}/{q.loc[0.9]:.1f}'
    )


if __name__ == '__main__':
    main()
