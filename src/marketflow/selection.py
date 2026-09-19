from __future__ import annotations

import numpy as np
import pandas as pd


def detect_opportunity_entries(
    scored_history: pd.DataFrame,
    short_threshold: float = 80.0,
    long_threshold: float = 80.0,
    min_sector_score: float = 50.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return names crossing into the short- and long-term momentum screens today.

    A name is a new entrant when its latest score is at/above the threshold and
    its previous valid cross-section observation was below the threshold (or did
    not exist). Sector strength is used as a confirmation gate, not as part of
    the momentum score itself.
    """
    if scored_history is None or scored_history.empty:
        empty = pd.DataFrame()
        return empty, empty

    x = scored_history.copy()
    x['date'] = pd.to_datetime(x['date']).dt.normalize()
    x = x.sort_values(['ticker', 'date'])
    latest_date = x['date'].max()

    for col in ['short_momentum_score', 'long_momentum_score']:
        if col not in x.columns:
            x[col] = np.nan

    x['prev_short_momentum_score'] = x.groupby('ticker')['short_momentum_score'].shift(1)
    x['prev_long_momentum_score'] = x.groupby('ticker')['long_momentum_score'].shift(1)
    today = x[x['date'].eq(latest_date)].copy()

    sector_ok = pd.to_numeric(today.get('sector_score'), errors='coerce').ge(float(min_sector_score))

    short_mask = (
        pd.to_numeric(today['short_momentum_score'], errors='coerce').ge(float(short_threshold))
        & (
            pd.to_numeric(today['prev_short_momentum_score'], errors='coerce').lt(float(short_threshold))
            | today['prev_short_momentum_score'].isna()
        )
        & sector_ok
    )
    long_mask = (
        pd.to_numeric(today['long_momentum_score'], errors='coerce').ge(float(long_threshold))
        & (
            pd.to_numeric(today['prev_long_momentum_score'], errors='coerce').lt(float(long_threshold))
            | today['prev_long_momentum_score'].isna()
        )
        & sector_ok
    )

    short_cols = [
        'date','ticker','sector','short_momentum_score','prev_short_momentum_score',
        'leadership_score','acceleration','flow_score','sector_score','stage',
    ]
    long_cols = [
        'date','ticker','sector','long_momentum_score','prev_long_momentum_score',
        'leadership_score','acceleration','trend_score','sector_score','stage',
    ]
    short = (today.loc[short_mask, [c for c in short_cols if c in today.columns]]
             .sort_values(['short_momentum_score','leadership_score'], ascending=False)
             .reset_index(drop=True))
    long = (today.loc[long_mask, [c for c in long_cols if c in today.columns]]
            .sort_values(['long_momentum_score','leadership_score'], ascending=False)
            .reset_index(drop=True))
    return short, long


def build_model_portfolio(
    latest: pd.DataFrame,
    size: int = 10,
    sector_cap: int = 2,
) -> pd.DataFrame:
    """Build a deterministic research portfolio from sector to stock.

    Step 1: rank sectors by current Sector Score.
    Step 2: rank stocks inside each sector by a composite that prioritizes
            stock leadership while retaining short/long momentum confirmation.
    Step 3: take one leader from the strongest sectors first, then fill remaining
            slots with the next-best candidates, capped per sector.

    This is a research basket, not an optimized portfolio or a recommendation.
    """
    if latest is None or latest.empty or size <= 0:
        return pd.DataFrame()

    x = latest.copy()
    required = ['leadership_score','sector_score','short_momentum_score','long_momentum_score','flow_score']
    for col in required:
        if col not in x.columns:
            x[col] = np.nan

    sectors = (x.groupby('sector', as_index=False)
               .agg(sector_score=('sector_score','median'),
                    sector_acceleration=('acceleration','median'),
                    sector_members=('ticker','count'))
               .sort_values(['sector_score','sector_acceleration'], ascending=False)
               .reset_index(drop=True))
    sectors['sector_rank'] = sectors.index + 1
    x = x.merge(sectors, on='sector', how='left', suffixes=('','_ranked'))

    # Sector quality is the first ranking layer; stock quality decides the
    # winner inside each sector.
    x['stock_selection_score'] = (
        0.45 * pd.to_numeric(x['leadership_score'], errors='coerce')
        + 0.20 * pd.to_numeric(x['short_momentum_score'], errors='coerce')
        + 0.20 * pd.to_numeric(x['long_momentum_score'], errors='coerce')
        + 0.15 * pd.to_numeric(x['flow_score'], errors='coerce')
    )
    x['portfolio_score'] = (
        0.45 * pd.to_numeric(x['sector_score_ranked'], errors='coerce')
        + 0.55 * x['stock_selection_score']
    )

    x = x.dropna(subset=['ticker','sector','portfolio_score'])
    x = x.sort_values(
        ['sector_rank','stock_selection_score','leadership_score'],
        ascending=[True,False,False],
    )

    selected = []
    counts: dict[str, int] = {}

    # First pass: strongest stock from each sector, preserving sector rank.
    first = x.groupby('sector', sort=False).head(1).sort_values('sector_rank')
    for _, row in first.iterrows():
        if len(selected) >= size:
            break
        sector = str(row['sector'])
        selected.append(row)
        counts[sector] = 1

    # Second pass: fill from best remaining opportunities, max N per sector.
    if len(selected) < size:
        selected_tickers = {str(r['ticker']) for r in selected}
        remaining = x[~x['ticker'].astype(str).isin(selected_tickers)].sort_values(
            ['portfolio_score','sector_rank','stock_selection_score'],
            ascending=[False,True,False],
        )
        for _, row in remaining.iterrows():
            if len(selected) >= size:
                break
            sector = str(row['sector'])
            if counts.get(sector, 0) >= sector_cap:
                continue
            selected.append(row)
            counts[sector] = counts.get(sector, 0) + 1

    if not selected:
        return pd.DataFrame()

    out = pd.DataFrame(selected).head(size).copy()
    out = out.sort_values(['portfolio_score','sector_rank'], ascending=[False,True]).reset_index(drop=True)
    out['portfolio_rank'] = out.index + 1
    out['weight'] = 100.0 / len(out)
    cols = [
        'portfolio_rank','ticker','sector','sector_rank','sector_score_ranked',
        'leadership_score','short_momentum_score','long_momentum_score',
        'flow_score','acceleration','portfolio_score','stage','weight',
    ]
    out = out[[c for c in cols if c in out.columns]].rename(
        columns={'sector_score_ranked':'sector_score'}
    )
    return out
