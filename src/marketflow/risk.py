from __future__ import annotations

import numpy as np
import pandas as pd


def _num(v, default=np.nan):
    try:
        out = float(v)
        return out if np.isfinite(out) else default
    except Exception:
        return default


def _clip(v: float, lo: float, hi: float) -> float:
    return float(min(hi, max(lo, v)))


def _strength_votes(row: pd.Series, cfg: dict) -> tuple[int, list[str]]:
    """Count independent signs that leadership / trend has broken."""
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


def _entry_risk_params(signal_row: pd.Series | None, cfg: dict) -> dict:
    """Convert the stock's pre-entry ATR into risk units.

    ATR is read from the signal-date row (known before the T+1 fill) so the
    adaptive stop does not use information from the fill session itself.
    """
    adaptive = bool(cfg.get('adaptive_volatility', True))
    base_stop = float(cfg.get('hard_stop_pct', 0.05))
    atr_pct = _num(signal_row.get('atr_pct_20')) if signal_row is not None else np.nan
    atr_regime = _num(signal_row.get('atr_regime_ratio'), 1.0) if signal_row is not None else 1.0

    if adaptive and np.isfinite(atr_pct) and atr_pct > 0:
        stop_pct = _clip(
            float(cfg.get('stop_atr_multiple', 2.0)) * atr_pct,
            float(cfg.get('min_stop_pct', 0.05)),
            float(cfg.get('max_stop_pct', 0.12)),
        )
    else:
        stop_pct = base_stop

    profit_arm = _clip(
        max(
            float(cfg.get('min_profit_arm_pct', 0.10)),
            float(cfg.get('profit_arm_r', 1.5)) * stop_pct,
        ),
        float(cfg.get('min_profit_arm_pct', 0.10)),
        float(cfg.get('max_profit_arm_pct', 0.22)),
    )
    profit_floor = _clip(
        max(
            float(cfg.get('min_profit_floor_pct', 0.02)),
            float(cfg.get('profit_floor_r', 0.30)) * stop_pct,
        ),
        float(cfg.get('min_profit_floor_pct', 0.02)),
        float(cfg.get('max_profit_floor_pct', 0.06)),
    )
    return {
        'adaptive': adaptive,
        'atr_pct': atr_pct,
        'atr_regime_ratio': atr_regime,
        'stop_pct': stop_pct,
        'profit_arm_pct': profit_arm,
        'profit_floor_pct': profit_floor,
    }


def _trail_pct(atr_pct: float, cfg: dict) -> float:
    """ATR-scaled trailing width; wider for high-volatility stocks."""
    base = float(cfg.get('peak_trail_pct', 0.07))
    if not bool(cfg.get('adaptive_volatility', True)) or not np.isfinite(atr_pct) or atr_pct <= 0:
        return base
    return _clip(
        float(cfg.get('trail_atr_multiple', 2.5)) * atr_pct,
        float(cfg.get('min_trail_pct', 0.08)),
        float(cfg.get('max_trail_pct', 0.18)),
    )


def simulate_position_lifecycle(
    price_history: pd.DataFrame,
    scored_history: pd.DataFrame,
    entries: pd.DataFrame,
    risk_cfg: dict | None = None,
) -> pd.DataFrame:
    """Simulate causal T+1 position management with stock-specific volatility.

    Entry signals are known after the signal-date close. The fill therefore
    occurs at the next session open. Initial risk is sized from ATR known on the
    signal date. Trailing stops use the previous completed session's ATR and
    ratchet upward only; they never loosen after being raised.
    """
    cfg = risk_cfg or {}
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
        ticker_px = px[px['ticker'].astype(str).eq(ticker)].sort_values('date').copy()
        signal_hist = ticker_px[ticker_px['date'].le(signal_date)]
        signal_row = signal_hist.iloc[-1] if not signal_hist.empty else None
        params = _entry_risk_params(signal_row, cfg)

        p = ticker_px[ticker_px['date'] > signal_date].copy()
        if p.empty:
            continue
        p = p.reset_index(drop=True)
        first = p.iloc[0]
        fill_price = _num(first.get('open'))
        if not np.isfinite(fill_price) or fill_price <= 0:
            fill_price = _num(first.get('close'))
        if not np.isfinite(fill_price) or fill_price <= 0:
            continue

        fill_date = pd.Timestamp(first['date'])
        stop_pct = params['stop_pct']
        profit_arm = params['profit_arm_pct']
        profit_floor = params['profit_floor_pct']
        hard_level = fill_price * (1.0 - stop_pct)

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
        ratchet_stop = hard_level
        latest_trail_pct = np.nan
        prev_atr_pct = params['atr_pct']

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

            # The active stop for today's bar was fully determined at the prior
            # close. This avoids using today's ATR/high/close to stop today's low.
            active_stop = ratchet_stop
            if np.isfinite(op) and op <= active_stop:
                exit_date = dt
                exit_price = op
                exit_reason = 'TRAILING_PROFIT_STOP_GAP' if armed else 'VOL_ADAPTIVE_STOP_GAP'
                status = 'CLOSED'
                break
            if np.isfinite(lo) and lo <= active_stop:
                exit_date = dt
                exit_price = active_stop
                exit_reason = 'TRAILING_PROFIT_STOP' if armed else 'VOL_ADAPTIVE_STOP'
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
            if armed and votes >= min_votes:
                pending_strength_exit = True
                pending_strength_reasons = reasons

            # Prepare tomorrow's stop from information known at today's close.
            bar_atr = _num(bar.get('atr_pct_20'), prev_atr_pct)
            if np.isfinite(bar_atr) and bar_atr > 0:
                prev_atr_pct = bar_atr
            if armed:
                latest_trail_pct = _trail_pct(prev_atr_pct, cfg)
                candidate = max(
                    fill_price * (1.0 + profit_floor),
                    peak_close * (1.0 - latest_trail_pct),
                )
                ratchet_stop = max(ratchet_stop, candidate)

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
            protective_stop = ratchet_stop
            if pending_strength_exit:
                status = 'EXIT_NEXT_OPEN'
                action = 'CUT / EXIT NEXT OPEN'
                exit_reason = 'STRENGTH_BREAK_PENDING: ' + ' + '.join(pending_strength_reasons[:3])
            elif armed:
                action = 'RIDE TREND' if latest_votes < min_votes else 'TIGHTEN'
            elif lifecycle_return <= -stop_pct * 0.70:
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
            'atr_pct_at_entry': params['atr_pct'],
            'atr_regime_ratio_at_entry': params['atr_regime_ratio'],
            'adaptive_stop_pct': stop_pct,
            'profit_arm_pct_used': profit_arm,
            'profit_floor_pct_used': profit_floor,
            'trail_pct_current': latest_trail_pct,
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
    stop_grid: tuple[float, ...] = (0.04,0.05,0.06,0.07,0.08,0.10,0.12),
) -> pd.DataFrame:
    """Fixed-stop sensitivity kept as a benchmark against adaptive risk."""
    if entries is None or entries.empty:
        return pd.DataFrame()
    base = simulate_position_lifecycle(
        price_history, scored_history, entries,
        {'adaptive_volatility': False, 'hard_stop_pct': 0.99, 'profit_arm_pct': 99.0,
         'min_profit_arm_pct': 99.0, 'max_profit_arm_pct': 99.0, 'strength_break_votes': 99}
    )
    baseline = base.set_index(['signal_date','ticker'])['no_stop_return'].to_dict() if not base.empty else {}

    rows = []
    for stop in stop_grid:
        sim = simulate_position_lifecycle(
            price_history, scored_history, entries,
            {'adaptive_volatility': False, 'hard_stop_pct': stop,
             'min_profit_arm_pct': 99.0, 'max_profit_arm_pct': 99.0, 'strength_break_votes': 99}
        )
        if sim.empty:
            continue
        vals = pd.to_numeric(sim['lifecycle_return'], errors='coerce')
        false_stops = 0
        rescued = 0
        stopped = 0
        for _, r in sim.iterrows():
            if 'STOP' in str(r.get('exit_reason','')):
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


def risk_policy_comparison(
    price_history: pd.DataFrame,
    scored_history: pd.DataFrame,
    entries: pd.DataFrame,
    adaptive_cfg: dict | None = None,
) -> pd.DataFrame:
    """Compare the old fixed rule with the adaptive ATR rule on identical trades."""
    if entries is None or entries.empty:
        return pd.DataFrame()
    policies = {
        'FIXED_5PCT': {
            'adaptive_volatility': False,
            'hard_stop_pct': 0.05,
            'peak_trail_pct': 0.07,
            'min_profit_arm_pct': 0.08,
            'max_profit_arm_pct': 0.08,
            'profit_arm_r': 0.0,
            'min_profit_floor_pct': 0.02,
            'max_profit_floor_pct': 0.02,
            'profit_floor_r': 0.0,
            'strength_break_votes': 2,
        },
        'ATR_ADAPTIVE': dict(adaptive_cfg or {}),
    }
    rows = []
    for name, cfg in policies.items():
        sim = simulate_position_lifecycle(price_history, scored_history, entries, cfg)
        if sim.empty:
            continue
        vals = pd.to_numeric(sim['lifecycle_return'], errors='coerce')
        wins = vals[vals > 0]
        losses = vals[vals <= 0]
        rows.append({
            'policy': name,
            'n': len(sim),
            'mean_return': vals.mean(),
            'median_return': vals.median(),
            'positive_rate': (vals > 0).mean(),
            'avg_winner': wins.mean() if len(wins) else np.nan,
            'avg_loser': losses.mean() if len(losses) else np.nan,
            'worst_trade': vals.min(),
            'closed_count': int(sim['status'].eq('CLOSED').sum()),
            'ride_trend_count': int(sim['action'].eq('RIDE TREND').sum()),
            'avg_stop_pct': pd.to_numeric(sim.get('adaptive_stop_pct'), errors='coerce').mean(),
        })
    return pd.DataFrame(rows)
