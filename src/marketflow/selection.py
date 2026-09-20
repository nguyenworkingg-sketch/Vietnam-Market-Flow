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



def build_entry_signal_history(
    scored_history: pd.DataFrame,
    min_sector_score: float = 50.0,
    min_leadership_score: float = 65.0,
    min_short_score: float = 65.0,
    min_long_score: float = 55.0,
    short_cross_threshold: float = 80.0,
    require_macd_positive: bool = True,
    require_ma_bull: bool = True,
    max_ma20_distance: float = 0.15,
    max_ret5: float = 0.18,
    ma_cross_bonus: float = 4.0,
    bb_breakout_bonus: float = 6.0,
) -> pd.DataFrame:
    """Create causal historical entry events.

    The signal is event-based, not a rank of whatever looks strongest today.
    A stock must pass quality and technical gates, must not be too extended, and
    must have a fresh trigger on that session.
    """
    if scored_history is None or scored_history.empty:
        return pd.DataFrame()

    x = scored_history.copy()
    x['date'] = pd.to_datetime(x['date']).dt.normalize()
    x = x.sort_values(['ticker','date']).reset_index(drop=True)

    needed = [
        'close','leadership_score','sector_score','short_momentum_score','long_momentum_score',
        'flow_score','trend_score','ret_5','ma20_distance','macd','macd_hist','macd_positive',
        'ma_bull','ma_cross_up','bb_breakout_after_squeeze',
    ]
    for col in needed:
        if col not in x.columns:
            x[col] = np.nan

    g = x.groupby('ticker', group_keys=False)
    x['prev_short_momentum_score'] = g['short_momentum_score'].shift(1)
    x['prev_macd_hist'] = g['macd_hist'].shift(1)

    short_cross = (
        pd.to_numeric(x['short_momentum_score'], errors='coerce').ge(float(short_cross_threshold))
        & (
            pd.to_numeric(x['prev_short_momentum_score'], errors='coerce').lt(float(short_cross_threshold))
            | x['prev_short_momentum_score'].isna()
        )
    )
    macd_turn = (
        pd.to_numeric(x['macd_hist'], errors='coerce').gt(0)
        & pd.to_numeric(x['prev_macd_hist'], errors='coerce').le(0)
        & pd.to_numeric(x['macd'], errors='coerce').gt(0)
        & pd.to_numeric(x['short_momentum_score'], errors='coerce').ge(max(70.0, float(min_short_score)))
    )
    ma_cross = x['ma_cross_up'].fillna(False).astype(bool)
    bb_breakout = x['bb_breakout_after_squeeze'].fillna(False).astype(bool)
    fresh_trigger = short_cross | macd_turn | ma_cross | bb_breakout

    quality = (
        pd.to_numeric(x['sector_score'], errors='coerce').ge(float(min_sector_score))
        & pd.to_numeric(x['leadership_score'], errors='coerce').ge(float(min_leadership_score))
        & pd.to_numeric(x['short_momentum_score'], errors='coerce').ge(float(min_short_score))
        & pd.to_numeric(x['long_momentum_score'], errors='coerce').ge(float(min_long_score))
    )
    if require_macd_positive:
        quality &= x['macd_positive'].fillna(False).astype(bool)
    if require_ma_bull:
        quality &= x['ma_bull'].fillna(False).astype(bool)

    anti_chase = (
        pd.to_numeric(x['ma20_distance'], errors='coerce').le(float(max_ma20_distance))
        & pd.to_numeric(x['ret_5'], errors='coerce').le(float(max_ret5))
    )

    signal = quality & anti_chase & fresh_trigger
    e = x.loc[signal].copy()
    if e.empty:
        return e

    base = (
        0.25 * pd.to_numeric(e['sector_score'], errors='coerce')
        + 0.25 * pd.to_numeric(e['leadership_score'], errors='coerce')
        + 0.20 * pd.to_numeric(e['short_momentum_score'], errors='coerce')
        + 0.15 * pd.to_numeric(e['long_momentum_score'], errors='coerce')
        + 0.10 * pd.to_numeric(e['flow_score'], errors='coerce')
        + 0.05 * pd.to_numeric(e['trend_score'], errors='coerce')
    )
    e['entry_score'] = base.fillna(0)
    e['entry_score'] += e['ma_cross_up'].fillna(False).astype(bool).astype(float) * float(ma_cross_bonus)
    e['entry_score'] += e['bb_breakout_after_squeeze'].fillna(False).astype(bool).astype(float) * float(bb_breakout_bonus)
    e['entry_score'] += short_cross.loc[e.index].astype(float) * 3.0

    e['entry_reason'] = np.select(
        [
            e['bb_breakout_after_squeeze'].fillna(False).astype(bool),
            e['ma_cross_up'].fillna(False).astype(bool),
            short_cross.loc[e.index],
            macd_turn.loc[e.index],
        ],
        ['BB squeeze breakout','MA20/MA50 cross','SM ngắn hạn vượt 80','MACD turn dương'],
        default='Fresh trend trigger',
    )
    e['entry_price'] = pd.to_numeric(e['close'], errors='coerce')
    e = e.rename(columns={'date':'entry_date'})
    cols = [
        'entry_date','ticker','sector','entry_price','entry_score','entry_reason',
        'sector_score','leadership_score','short_momentum_score','long_momentum_score',
        'flow_score','trend_score','ret_5','ma20_distance','macd','macd_hist',
        'ma20','ma50','stage',
    ]
    return e[[col for col in cols if col in e.columns]].sort_values(
        ['entry_date','entry_score'], ascending=[True,False]
    ).reset_index(drop=True)


def build_entry_candidates(
    scored_history: pd.DataFrame,
    top_n: int = 3,
    min_sector_score: float = 50.0,
    min_leadership_score: float = 65.0,
    min_short_score: float = 65.0,
    min_long_score: float = 55.0,
    short_cross_threshold: float = 80.0,
    require_macd_positive: bool = True,
    require_ma_bull: bool = True,
    max_ma20_distance: float = 0.15,
    max_ret5: float = 0.18,
    max_age_sessions: int = 3,
    max_distance_from_entry: float = 0.08,
    ma_cross_bonus: float = 4.0,
    bb_breakout_bonus: float = 6.0,
) -> pd.DataFrame:
    """Return only fresh entry opportunities, never stale momentum leaders.

    The most recent historical entry event is attached to the latest session.
    A setup expires after the configured age or once price is too far above the
    original entry reference price.
    """
    if scored_history is None or scored_history.empty or top_n <= 0:
        return pd.DataFrame()

    x = scored_history.copy()
    x['date'] = pd.to_datetime(x['date']).dt.normalize()
    x = x.sort_values(['ticker','date']).reset_index(drop=True)
    latest_date = x['date'].max()
    latest = x[x['date'].eq(latest_date)].copy()

    events = build_entry_signal_history(
        x,
        min_sector_score=min_sector_score,
        min_leadership_score=min_leadership_score,
        min_short_score=min_short_score,
        min_long_score=min_long_score,
        short_cross_threshold=short_cross_threshold,
        require_macd_positive=require_macd_positive,
        require_ma_bull=require_ma_bull,
        max_ma20_distance=max_ma20_distance,
        max_ret5=max_ret5,
        ma_cross_bonus=ma_cross_bonus,
        bb_breakout_bonus=bb_breakout_bonus,
    )
    if events.empty:
        return pd.DataFrame()

    recent_events = events.sort_values(['ticker','entry_date']).groupby('ticker', as_index=False).tail(1)
    merged = latest.merge(recent_events, on=['ticker','sector'], how='inner', suffixes=('_current','_entry'))
    if merged.empty:
        return pd.DataFrame()

    age_map = {}
    for ticker, g in x.groupby('ticker'):
        dates = g['date'].drop_duplicates().sort_values().tolist()
        pos = {pd.Timestamp(d): i for i,d in enumerate(dates)}
        age_map[str(ticker)] = (pos, len(dates)-1)

    ages=[]
    for _, row in merged.iterrows():
        pos,last_i = age_map.get(str(row['ticker']), ({},0))
        entry_i = pos.get(pd.Timestamp(row['entry_date']), last_i)
        ages.append(max(0, last_i-entry_i))
    merged['entry_age_sessions'] = ages
    merged['current_price'] = pd.to_numeric(merged['close_current'], errors='coerce')
    merged['since_entry_pct'] = merged['current_price'] / pd.to_numeric(merged['entry_price'], errors='coerce') - 1

    current_ok = (
        pd.to_numeric(merged['sector_score_current'], errors='coerce').ge(float(min_sector_score))
        & pd.to_numeric(merged['ma20_distance_current'], errors='coerce').le(float(max_ma20_distance))
        & pd.to_numeric(merged['ret_5_current'], errors='coerce').le(float(max_ret5))
        & merged['entry_age_sessions'].le(int(max_age_sessions))
        & merged['since_entry_pct'].le(float(max_distance_from_entry))
    )
    if require_macd_positive:
        current_ok &= merged['macd_positive_current'].fillna(False).astype(bool)
    if require_ma_bull:
        current_ok &= merged['ma_bull_current'].fillna(False).astype(bool)

    out = merged.loc[current_ok].copy()
    if out.empty:
        return pd.DataFrame()

    out['fresh_entry_score'] = (
        pd.to_numeric(out['entry_score'], errors='coerce')
        - 1.5 * out['entry_age_sessions']
        - 35.0 * out['since_entry_pct'].clip(lower=0)
    )
    out = out.sort_values(
        ['fresh_entry_score','entry_score','sector_score_current','leadership_score_current'],
        ascending=False,
    ).head(top_n).reset_index(drop=True)
    out['entry_rank'] = out.index + 1
    out['macd_status'] = 'DƯƠNG'
    out['ma_status'] = 'MA20 > MA50'
    out['technical_setup'] = out['entry_reason']

    cols = [
        'entry_rank','ticker','sector','entry_date','entry_price','current_price',
        'since_entry_pct','entry_age_sessions','fresh_entry_score','entry_score','entry_reason',
        'sector_score_current','leadership_score_current','short_momentum_score_current',
        'long_momentum_score_current','flow_score_current','trend_score_current',
        'macd_status','ma_status','stage_current',
    ]
    out = out[[col for col in cols if col in out.columns]].rename(columns={
        'fresh_entry_score':'entry_score_current',
        'sector_score_current':'sector_score',
        'leadership_score_current':'leadership_score',
        'short_momentum_score_current':'short_momentum_score',
        'long_momentum_score_current':'long_momentum_score',
        'flow_score_current':'flow_score',
        'trend_score_current':'trend_score',
        'stage_current':'stage',
    })
    return out
