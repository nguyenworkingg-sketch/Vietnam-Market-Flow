from __future__ import annotations

import numpy as np
import pandas as pd


def _num(v, default=np.nan):
    try:
        out = float(v)
        return out if np.isfinite(out) else default
    except Exception:
        return default


def _strength_votes(row: pd.Series, cfg: dict) -> tuple[int, list[str]]:
    """Count independent signs that the leadership/price trend has broken."""
    reasons: list[str] = []
    rs = _num(row.get('real_strength_score'))
    leadership = _num(row.get('leadership_score'))
    trend = _num(row.get('trend_score'))
    close = _num(row.get('close'))
    ma20 = _num(row.get('ma20'))
    ma20_slope = _num(row.get('ma20_slope_5'))

    if np.isfinite(rs) and rs < float(cfg.get('strength_real_floor', 60)):
        reasons.append('Real Strength < floor')
    if np.isfinite(leadership) and leadership < float(cfg.get('strength_leadership_floor', 60)):
        reasons.append('Leadership < floor')
    if np.isfinite(trend) and trend < float(cfg.get('strength_trend_floor', 55)):
        reasons.append('Trend score < floor')
    if np.isfinite(close) and np.isfinite(ma20) and np.isfinite(ma20_slope):
        if close < ma20 and ma20_slope <= 0:
            reasons.append('Close < MA20 & MA20 slope <= 0')
    weekly = row.get('weekly_trend_confirm')
    if pd.notna(weekly) and not bool(weekly):
        reasons.append('Weekly trend broken')
    return len(reasons), reasons


def simulate_position_lifecycle(
    price_history: pd.DataFrame,
    scored_history: pd.DataFrame,
    entries: pd.DataFrame,
    risk_cfg: dict | None = None,
) -> pd.DataFrame:
    """Simulate causal position management from each entry event.

    Entry signals are assumed known only after the signal-date close, therefore
    fills happen at the next available session open. Hard/trailing stops use
    same-session open/low with gap handling. A strength-break decision observed
    at a close exits at the next session open to avoid look-ahead.
    """
    cfg = risk_cfg or {}
    hard_stop = float(cfg.get('hard_stop_pct', 0.05))
    profit_arm = float(cfg.get('profit_arm_pct', 0.08))
    profit_floor = float(cfg.get('profit_floor_pct', 0.02))
    peak_trail = float(cfg.get('peak_trail_pct', 0.07))
    min_votes = int(cfg.get('strength_break_votes', 2))

    if entries is None or entries.empty or price_history is None or price_history.empty:
        return pd.DataFrame()

    px = price_history.copy()
    px['date'] = pd.to_datetime(px['date']).dt.normalize()
    px = px.sort_values(['ticker','date']).drop_duplicates(['ticker','date'], keep='last')

    scores = scored_history.copy() if scored_history is not None else pd.DataFrame()
    if not scores.empty:
        scores['date'] = pd.to_datetime(scores['date']).dt.normalize()
        keep = [
            'date','ticker','real_strength_score','leadership_score','trend_score',
            'short_momentum_score','long_momentum_score','sector_score','stage'
        ]
        scores = scores[[c for c in keep if c in scores.columns]].drop_duplicates(['date','ticker'], keep='last')
        px = px.merge(scores, on=['date','ticker'], how='left', suffixes=('','_score'))

    e = entries.copy()
    date_col = 'entry_date' if 'entry_date' in e.columns else 'signal_date'
    e[date_col] = pd.to_datetime(e[date_col]).dt.normalize()

    rows = []
    for _, ent in e.iterrows():
        ticker = str(ent.get('ticker',''))
        signal_date = pd.Timestamp(ent[date_col])
        p = px[(px['ticker'].astype(str) == ticker) & (px['date'] > signal_date)].copy()
        if p.empty:
            continue
        p = p.sort_values('date').reset_index(drop=True)
        first = p.iloc[0]
        fill_price = _num(first.get('open'))
        if not np.isfinite(fill_price) or fill_price <= 0:
            fill_price = _num(first.get('close'))
        if not np.isfinite(fill_price) or fill_price <= 0:
            continue
        fill_date = pd.Timestamp(first['date'])
        hard_level = fill_price * (1.0 - hard_stop)

        peak_close = fill_price
        peak_date = fill_date
        armed = False
        pending_strength_exit = False
        pending_strength_reasons: list[str] = []
        exit_date = pd.NaT
        exit_price = np.nan
        exit_reason = ''
        status = 'OPEN'
        latest_votes = 0
        latest_reasons: list[str] = []

        for _, bar in p.iterrows():
            dt = pd.Timestamp(bar['date'])
            op = _num(bar.get('open'))
            lo = _num(bar.get('low'))
            cl = _num(bar.get('close'))

            if pending_strength_exit:
                exit_date = dt
                exit_price = op if np.isfinite(op) and op > 0 else cl
                exit_reason = 'STRENGTH_BREAK'
                if pending_strength_reasons:
                    exit_reason += ': ' + ' + '.join(pending_strength_reasons[:3])
                status = 'CLOSED'
                break

            # Stop levels are based only on information known before this bar.
            trailing_level = np.nan
            if armed:
                trailing_level = max(
                    fill_price * (1.0 + profit_floor),
                    peak_close * (1.0 - peak_trail),
                )
            active_stop = hard_level if not armed else max(hard_level, trailing_level)

            if np.isfinite(op) and op <= active_stop:
                exit_date = dt
                exit_price = op
                exit_reason = 'TRAILING_PROFIT_STOP_GAP' if armed else 'HARD_STOP_GAP'
                status = 'CLOSED'
                break
            if np.isfinite(lo) and lo <= active_stop:
                exit_date = dt
                exit_price = active_stop
                exit_reason = 'TRAILING_PROFIT_STOP' if armed else 'HARD_STOP'
                status = 'CLOSED'
                break

            if np.isfinite(cl) and cl > peak_close:
                peak_close = cl
                peak_date = dt
            peak_return = peak_close / fill_price - 1.0
            if peak_return >= profit_arm:
                armed = True

            votes, reasons = _strength_votes(bar, cfg)
            latest_votes, latest_reasons = votes, reasons
            # Once a position has proven itself, exit on a multi-factor strength
            # break at the next open instead of allowing a round-trip to entry.
            if armed and votes >= min_votes:
                pending_strength_exit = True
                pending_strength_reasons = reasons

        last = p.iloc[-1]
        latest_close = _num(last.get('close'))
        latest_date = pd.Timestamp(last['date'])
        peak_return = peak_close / fill_price - 1.0

        if status == 'CLOSED':
            lifecycle_return = exit_price / fill_price - 1.0
            mark_price = exit_price
            mark_date = exit_date
            action = 'CLOSED'
            protective_stop = np.nan
            drawdown_from_peak = exit_price / peak_close - 1.0
        else:
            lifecycle_return = latest_close / fill_price - 1.0
            mark_price = latest_close
            mark_date = latest_date
            drawdown_from_peak = latest_close / peak_close - 1.0
            protective_stop = (
                max(hard_level, fill_price*(1.0+profit_floor), peak_close*(1.0-peak_trail))
                if armed else hard_level
            )
            if pending_strength_exit:
                status = 'EXIT_NEXT_OPEN'
                action = 'CUT / EXIT NEXT OPEN'
                exit_reason = 'STRENGTH_BREAK_PENDING: ' + ' + '.join(pending_strength_reasons[:3])
            elif armed:
                action = 'RIDE TREND' if latest_votes < min_votes else 'TIGHTEN'
            elif lifecycle_return <= -hard_stop * 0.7:
                action = 'RISK WATCH'
            else:
                action = 'HOLD'

        no_stop_return = latest_close / fill_price - 1.0
        rows.append({
            'signal_date': signal_date,
            'ticker': ticker,
            'sector': ent.get('sector',''),
            'entry_reason': ent.get('entry_reason',''),
            'model_version': ent.get('model_version',''),
            'fill_date': fill_date,
            'fill_price': fill_price,
            'status': status,
            'action': action,
            'exit_date': exit_date,
            'exit_price': exit_price,
            'exit_reason': exit_reason,
            'mark_date': mark_date,
            'mark_price': mark_price,
            'lifecycle_return': lifecycle_return,
            'no_stop_return': no_stop_return,
            'peak_close': peak_close,
            'peak_date': peak_date,
            'peak_return': peak_return,
            'drawdown_from_peak': drawdown_from_peak,
            'hard_stop_level': hard_level,
            'protective_stop': protective_stop,
            'strength_break_votes': latest_votes,
            'strength_break_reasons': ' · '.join(latest_reasons),
            'profit_protection_armed': armed,
        })

    return pd.DataFrame(rows)


def stop_sensitivity_study(
    price_history: pd.DataFrame,
    scored_history: pd.DataFrame,
    entries: pd.DataFrame,
    stop_grid: tuple[float, ...] = (0.04,0.05,0.06,0.07,0.08,0.10),
) -> pd.DataFrame:
    """Test hard-stop sensitivity without pretending a tiny sample is optimal."""
    if entries is None or entries.empty:
        return pd.DataFrame()
    # Baseline mark-to-market path with an effectively disabled stop.
    base = simulate_position_lifecycle(
        price_history, scored_history, entries,
        {'hard_stop_pct': 0.99, 'profit_arm_pct': 99.0, 'strength_break_votes': 99}
    )
    baseline = base.set_index(['signal_date','ticker'])['no_stop_return'].to_dict() if not base.empty else {}

    rows = []
    for stop in stop_grid:
        sim = simulate_position_lifecycle(
            price_history, scored_history, entries,
            {'hard_stop_pct': stop, 'profit_arm_pct': 99.0, 'strength_break_votes': 99}
        )
        if sim.empty:
            continue
        vals = pd.to_numeric(sim['lifecycle_return'], errors='coerce')
        false_stops = 0
        rescued = 0
        stopped = 0
        for _, r in sim.iterrows():
            if str(r.get('exit_reason','')).startswith('HARD_STOP'):
                stopped += 1
                key = (pd.Timestamp(r['signal_date']), str(r['ticker']))
                end_ret = baseline.get(key, np.nan)
                if pd.notna(end_ret) and end_ret > 0:
                    false_stops += 1
                if pd.notna(end_ret) and end_ret <= 0:
                    rescued += 1
        rows.append({
            'hard_stop_pct': stop,
            'n': len(sim),
            'mean_return': vals.mean(),
            'median_return': vals.median(),
            'positive_rate': (vals > 0).mean(),
            'stopped_count': stopped,
            'false_stop_winners': false_stops,
            'rescued_losers': rescued,
        })
    return pd.DataFrame(rows)
