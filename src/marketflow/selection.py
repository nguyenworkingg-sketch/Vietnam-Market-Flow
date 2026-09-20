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
    min_sector_score: float = 55.0,
    min_leadership_score: float = 70.0,
    min_short_score: float = 65.0,
    min_long_score: float = 70.0,
    require_macd_positive: bool = True,
    require_medium_trend: bool = True,
    require_weekly_trend: bool = True,
    max_ma20_distance: float = 0.08,
    max_ret5: float = 0.10,
    squeeze_bonus: float = 5.0,
    pullback_bonus: float = 3.0,
) -> pd.DataFrame:
    """Generate sparse, causal "real strength" entry events.

    The model deliberately separates *strength* from *entry timing*:
    1) Strength must persist across medium and weekly trend frames and beat both
       benchmark and sector over the larger windows.
    2) Entry timing waits for either a controlled pullback/resumption near MA20
       or a breakout from a recent Bollinger squeeze.
    3) MA20/MA50 crosses and one-day momentum threshold crosses are NOT entry
       triggers because they are too lagging/chase-prone on their own.
    """
    if scored_history is None or scored_history.empty:
        return pd.DataFrame()

    x = scored_history.copy()
    x['date'] = pd.to_datetime(x['date']).dt.normalize()
    x = x.sort_values(['ticker','date']).reset_index(drop=True)

    needed = [
        'close','leadership_score','sector_score','short_momentum_score','long_momentum_score',
        'flow_score','trend_score','ret_5','ma20','ma20_distance','ma50',
        'macd','macd_hist','macd_positive','volume_ratio_20',
        'bb_squeeze_recent_10','bb_breakout_after_squeeze','prior_high_10',
        'medium_trend_confirm','weekly_trend_confirm','weekly_ret12',
        'rs_60','rs_120','rs_sector_60','sector_rs_60',
    ]
    for col in needed:
        if col not in x.columns:
            x[col] = np.nan

    g = x.groupby('ticker', group_keys=False)
    x['prev_close'] = g['close'].shift(1)
    x['prev_ma20'] = g['ma20'].shift(1)
    x['prev_macd_hist'] = g['macd_hist'].shift(1)
    x['leadership_med10'] = g['leadership_score'].transform(
        lambda s: s.rolling(10, min_periods=5).median()
    )
    x['long_med10'] = g['long_momentum_score'].transform(
        lambda s: s.rolling(10, min_periods=5).median()
    )
    x['sector_med10'] = g['sector_score'].transform(
        lambda s: s.rolling(10, min_periods=5).median()
    )

    persistent_strength = (
        pd.to_numeric(x['leadership_med10'], errors='coerce').ge(float(min_leadership_score))
        & pd.to_numeric(x['long_med10'], errors='coerce').ge(float(min_long_score))
        & pd.to_numeric(x['sector_med10'], errors='coerce').ge(float(min_sector_score))
        & pd.to_numeric(x['short_momentum_score'], errors='coerce').ge(float(min_short_score))
        & pd.to_numeric(x['rs_60'], errors='coerce').gt(0)
        & pd.to_numeric(x['rs_120'], errors='coerce').gt(0)
        & pd.to_numeric(x['rs_sector_60'], errors='coerce').gt(0)
        & pd.to_numeric(x['sector_rs_60'], errors='coerce').gt(0)
        & pd.to_numeric(x['weekly_ret12'], errors='coerce').gt(0)
    )
    if require_medium_trend:
        persistent_strength &= x['medium_trend_confirm'].fillna(False).astype(bool)
    if require_weekly_trend:
        persistent_strength &= x['weekly_trend_confirm'].fillna(False).astype(bool)
    if require_macd_positive:
        persistent_strength &= (
            pd.to_numeric(x['macd'], errors='coerce').gt(0)
            & pd.to_numeric(x['macd_hist'], errors='coerce').gt(0)
        )

    # Entry A: resume after a controlled pullback near MA20. This is designed to
    # enter after strength is already proven, but before price becomes extended.
    pullback_resume = (
        pd.to_numeric(x['prev_close'], errors='coerce').le(
            pd.to_numeric(x['prev_ma20'], errors='coerce') * 1.02
        )
        & pd.to_numeric(x['close'], errors='coerce').gt(pd.to_numeric(x['ma20'], errors='coerce'))
        & pd.to_numeric(x['macd_hist'], errors='coerce').gt(0)
        & pd.to_numeric(x['macd_hist'], errors='coerce').gt(
            pd.to_numeric(x['prev_macd_hist'], errors='coerce')
        )
        & pd.to_numeric(x['volume_ratio_20'], errors='coerce').ge(1.0)
        & pd.to_numeric(x['ma20_distance'], errors='coerce').between(0, min(0.06, float(max_ma20_distance)))
        & pd.to_numeric(x['ret_5'], errors='coerce').between(-0.05, min(0.08, float(max_ret5)))
    )

    # Entry B: breakout from genuine compression, but only while still close to
    # MA20. This is stricter than "price is strong today".
    squeeze_breakout = (
        x['bb_squeeze_recent_10'].fillna(False).astype(bool)
        & pd.to_numeric(x['close'], errors='coerce').gt(pd.to_numeric(x['prior_high_10'], errors='coerce'))
        & pd.to_numeric(x['volume_ratio_20'], errors='coerce').ge(1.20)
        & pd.to_numeric(x['ma20_distance'], errors='coerce').between(0, float(max_ma20_distance))
        & pd.to_numeric(x['ret_5'], errors='coerce').le(float(max_ret5))
    )

    anti_chase = (
        pd.to_numeric(x['ma20_distance'], errors='coerce').between(-0.02, float(max_ma20_distance))
        & pd.to_numeric(x['ret_5'], errors='coerce').between(-0.06, float(max_ret5))
    )
    raw_signal = persistent_strength & anti_chase & (pullback_resume | squeeze_breakout)

    # One entry per meaningful trend leg. A brief dip below MA20 is a pullback,
    # not a reset. Re-entry requires a larger-frame deterioration first.
    reset = (
        (pd.to_numeric(x['close'], errors='coerce') < pd.to_numeric(x['ma50'], errors='coerce'))
        | (~x['weekly_trend_confirm'].fillna(False).astype(bool))
        | pd.to_numeric(x['leadership_med10'], errors='coerce').lt(55)
        | pd.to_numeric(x['long_med10'], errors='coerce').lt(55)
    )
    accepted = pd.Series(False, index=x.index)
    for _, idxs in x.groupby('ticker', sort=False).groups.items():
        active = False
        for idx in idxs:
            if active and bool(reset.loc[idx]):
                active = False
            if (not active) and bool(raw_signal.loc[idx]):
                accepted.loc[idx] = True
                active = True

    e = x.loc[accepted].copy()
    if e.empty:
        return e

    e['entry_score'] = (
        0.25 * pd.to_numeric(e['leadership_med10'], errors='coerce')
        + 0.25 * pd.to_numeric(e['long_med10'], errors='coerce')
        + 0.20 * pd.to_numeric(e['sector_med10'], errors='coerce')
        + 0.10 * pd.to_numeric(e['short_momentum_score'], errors='coerce')
        + 0.10 * pd.to_numeric(e['flow_score'], errors='coerce')
        + 0.10 * pd.to_numeric(e['trend_score'], errors='coerce')
    ).fillna(0)
    e['entry_score'] += squeeze_breakout.loc[e.index].astype(float) * float(squeeze_bonus)
    e['entry_score'] += pullback_resume.loc[e.index].astype(float) * float(pullback_bonus)
    e['entry_reason'] = np.where(
        squeeze_breakout.loc[e.index],
        'Squeeze breakout + HTF confirm',
        'Pullback resume + HTF confirm',
    )
    e['entry_price'] = pd.to_numeric(e['close'], errors='coerce')
    e = e.rename(columns={'date':'entry_date'})
    cols = [
        'entry_date','ticker','sector','entry_price','entry_score','entry_reason',
        'leadership_med10','long_med10','sector_med10','short_momentum_score',
        'flow_score','trend_score','rs_60','rs_120','rs_sector_60','sector_rs_60',
        'weekly_ret12','medium_trend_confirm','weekly_trend_confirm',
        'ret_5','ma20_distance','volume_ratio_20','macd','macd_hist','ma20','ma50','stage',
    ]
    return e[[col for col in cols if col in e.columns]].sort_values(
        ['entry_date','entry_score'], ascending=[True,False]
    ).reset_index(drop=True)


def build_entry_candidates(
    scored_history: pd.DataFrame,
    top_n: int = 3,
    min_sector_score: float = 55.0,
    min_leadership_score: float = 70.0,
    min_short_score: float = 65.0,
    min_long_score: float = 70.0,
    require_macd_positive: bool = True,
    require_medium_trend: bool = True,
    require_weekly_trend: bool = True,
    max_ma20_distance: float = 0.08,
    max_ret5: float = 0.10,
    max_age_sessions: int = 2,
    max_distance_from_entry: float = 0.05,
    squeeze_bonus: float = 5.0,
    pullback_bonus: float = 3.0,
) -> pd.DataFrame:
    """Return only fresh v3 entries that are still close to the original trigger."""
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
        require_macd_positive=require_macd_positive,
        require_medium_trend=require_medium_trend,
        require_weekly_trend=require_weekly_trend,
        max_ma20_distance=max_ma20_distance,
        max_ret5=max_ret5,
        squeeze_bonus=squeeze_bonus,
        pullback_bonus=pullback_bonus,
    )
    if events.empty:
        return pd.DataFrame()

    recent_events = events.sort_values(['ticker','entry_date']).groupby('ticker', as_index=False).tail(1)
    current_cols = [
        'ticker','sector','close','sector_score','leadership_score','short_momentum_score',
        'long_momentum_score','flow_score','trend_score','ma20_distance','ret_5',
        'medium_trend_confirm','weekly_trend_confirm','stage',
    ]
    current = latest[[col for col in current_cols if col in latest.columns]].copy().rename(columns={
        'close':'current_price',
        'sector_score':'sector_score_current',
        'leadership_score':'leadership_score_current',
        'short_momentum_score':'short_momentum_score_current',
        'long_momentum_score':'long_momentum_score_current',
        'flow_score':'flow_score_current',
        'trend_score':'trend_score_current',
        'ma20_distance':'ma20_distance_current',
        'ret_5':'ret_5_current',
        'medium_trend_confirm':'medium_trend_confirm_current',
        'weekly_trend_confirm':'weekly_trend_confirm_current',
        'stage':'stage_current',
    })
    merged = current.merge(recent_events, on=['ticker','sector'], how='inner')
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
    merged['current_price'] = pd.to_numeric(merged['current_price'], errors='coerce')
    merged['since_entry_pct'] = merged['current_price'] / pd.to_numeric(merged['entry_price'], errors='coerce') - 1

    current_ok = (
        merged['entry_age_sessions'].le(int(max_age_sessions))
        & merged['since_entry_pct'].between(-0.06, float(max_distance_from_entry))
        & pd.to_numeric(merged['ma20_distance_current'], errors='coerce').le(float(max_ma20_distance))
        & pd.to_numeric(merged['ret_5_current'], errors='coerce').le(float(max_ret5))
    )
    if require_medium_trend:
        current_ok &= merged['medium_trend_confirm_current'].fillna(False).astype(bool)
    if require_weekly_trend:
        current_ok &= merged['weekly_trend_confirm_current'].fillna(False).astype(bool)

    out = merged.loc[current_ok].copy()
    if out.empty:
        return pd.DataFrame()

    out['fresh_entry_score'] = (
        pd.to_numeric(out['entry_score'], errors='coerce')
        - 2.0 * out['entry_age_sessions']
        - 45.0 * out['since_entry_pct'].clip(lower=0)
    )
    out = out.sort_values(
        ['fresh_entry_score','entry_score','sector_score_current','leadership_score_current'],
        ascending=False,
    ).head(top_n).reset_index(drop=True)
    out['entry_rank'] = out.index + 1
    out['technical_setup'] = out['entry_reason']

    cols = [
        'entry_rank','ticker','sector','entry_date','entry_price','current_price',
        'since_entry_pct','entry_age_sessions','fresh_entry_score','entry_score','entry_reason',
        'leadership_med10','long_med10','sector_med10',
        'sector_score_current','leadership_score_current','short_momentum_score_current',
        'long_momentum_score_current','flow_score_current','trend_score_current',
        'medium_trend_confirm_current','weekly_trend_confirm_current','stage_current',
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
