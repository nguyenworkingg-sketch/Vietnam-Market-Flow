from __future__ import annotations

import math
from pathlib import Path
import pandas as pd

from .features import add_stock_features, add_sector_features
from .scoring import build_scores, add_stage
from .regime import market_regime
from .storage import DuckStore
from .dashboard import render_dashboard
from .selection import detect_opportunity_entries, build_model_portfolio, build_entry_candidates, build_entry_signal_history
from .supabase_store import SupabaseRESTStore


def _valid_cross_section_dates(feat: pd.DataFrame, eligible: pd.Series, model: dict):
    counts = (feat.loc[eligible]
              .groupby('date')['ticker'].nunique()
              .sort_index())
    if counts.empty:
        return pd.Index([]), counts, 0

    lookback = int(model.get('coverage_reference_days', 30))
    recent = counts.tail(lookback)
    reference = int(recent.max())
    min_n = max(
        int(model.get('min_cross_section_stocks', 50)),
        int(math.ceil(reference * float(model.get('min_cross_section_coverage', 0.70)))),
    )
    valid_dates = counts[counts >= min_n].index
    return valid_dates, counts, reference


def run(provider, cfg: dict, root: str | Path, history_start: str | None = None, provider_name: str = 'unknown'):
    root = Path(root)
    model = cfg['model']
    start = history_start or model['history_start']
    sb = SupabaseRESTStore.from_env()
    run_id = None
    if sb is not None:
        try:
            run_id = sb.start_run(provider_name)
        except Exception as exc:
            print(f'[WARN] Could not log Supabase run start: {exc}')

    try:
        universe = provider.get_universe().copy()
        universe['ticker'] = universe['ticker'].astype(str).str.upper()
        bench = provider.get_benchmark(model['benchmark'], start)

        # Vnstock Community is reliable enough for a one-time historical
        # bootstrap, but refetching 12+ months for hundreds of stocks every day
        # is both slow and quota-fragile. Persist the historical panel in
        # Supabase and, after bootstrap, append only the bulk price-board bar.
        if (
            sb is not None
            and provider_name == 'vnstock'
            and hasattr(provider, 'select_live_symbols')
            and hasattr(provider, 'get_board_bars')
        ):
            symbols = provider.select_live_symbols(universe)
            cache = sb.fetch_ohlcv(start, tickers=symbols)
            counts = cache.groupby('ticker')['date'].nunique() if not cache.empty else pd.Series(dtype='int64')
            required = int(model['min_history_days']) + 30
            missing = [s for s in symbols if int(counts.get(s, 0)) < required]

            if missing:
                print(f'[CACHE] Bootstrap/repair histories: {len(missing)} symbols')
                fresh = provider.get_prices_for_symbols(missing, start)
                sb.sync_ohlcv(fresh)
                cache = pd.concat([cache, fresh], ignore_index=True)

            as_of = pd.to_datetime(bench['date']).max()
            board = provider.get_board_bars(symbols, as_of)
            if not board.empty:
                sb.sync_ohlcv(board)
                cache = pd.concat([cache, board], ignore_index=True)

            prices = (cache[cache['ticker'].isin(symbols)]
                      .drop_duplicates(['date','ticker'], keep='last')
                      .sort_values(['ticker','date'])
                      .reset_index(drop=True))
            if prices.empty:
                raise RuntimeError('OHLCV cache is empty after bootstrap/update.')
            print(
                f'[CACHE] prices={len(prices):,} rows, '
                f'symbols={prices["ticker"].nunique()}, '
                f'asof={pd.to_datetime(prices["date"]).max().date()}'
            )
        else:
            prices = provider.get_prices(start)

        meta_cols = ['ticker', 'sector'] + (['exchange'] if 'exchange' in universe.columns else [])
        prices = prices.merge(universe[meta_cols].drop_duplicates('ticker'), on='ticker', how='left')
        feat = add_stock_features(prices, bench)
        feat = add_sector_features(feat)

        feat['history_n'] = feat.groupby('ticker').cumcount() + 1
        eligible = (
            (feat['close'] >= model['min_price']) &
            (feat['value_avg_20'] >= model['min_avg_value_20']) &
            (feat['history_n'] >= model['min_history_days']) &
            feat['sector'].notna()
        )

        valid_dates, coverage_counts, coverage_reference = _valid_cross_section_dates(feat, eligible, model)
        if len(valid_dates) == 0:
            diag_cols = [c for c in ['ticker','date','close','volume','value','value_avg_20','history_n','sector'] if c in feat.columns]
            diag = (feat.sort_values('date').groupby('ticker', as_index=False).tail(1)[diag_cols]
                    .sort_values('ticker'))
            print('[ELIGIBILITY DIAGNOSTIC]')
            print(diag.to_string(index=False))
            raise RuntimeError('No sufficiently complete cross-section after filters; inspect provider coverage.')

        # Never rank a partial/intraday cross-section against itself.  Scores,
        # acceleration and market breadth are calculated only on dates whose
        # coverage is representative of the recent universe.
        scoring_mask = eligible & feat['date'].isin(valid_dates)
        scored = build_scores(feat.loc[scoring_mask].copy(), cfg)
        if scored.empty:
            raise RuntimeError('No eligible stocks after cross-section coverage gate.')
        scored = add_stage(scored, cfg)

        regime = market_regime(feat.loc[scoring_mask].copy(), bench)
        coverage = coverage_counts.rename('coverage_stocks').reset_index()
        regime = regime.merge(coverage, on='date', how='left')
        regime['coverage_reference'] = coverage_reference

        latest_date = scored['date'].max()
        latest = scored[scored['date'] == latest_date].copy()
        latest_coverage = int(coverage_counts.get(latest_date, len(latest)))
        print(
            f'[COVERAGE] asof={pd.Timestamp(latest_date).date()} '
            f'stocks={latest_coverage} reference={coverage_reference}'
        )

        reg_latest = regime[regime['date'] == latest_date]
        reg_latest = reg_latest.iloc[-1].to_dict() if not reg_latest.empty else {}

        out = root / 'outputs'
        out.mkdir(exist_ok=True)
        latest.sort_values('leadership_score', ascending=False).to_csv(out / 'scores_latest.csv', index=False)
        hist = scored[['date','ticker','sector','leadership_score','short_momentum_score','long_momentum_score','acceleration','rs_score','flow_score','trend_score','sector_score','stage']]
        try:
            hist.to_parquet(out / 'scores_history.parquet', index=False)
        except Exception:
            hist.to_csv(out / 'scores_history.csv', index=False)

        # Research panel keeps the full price path, including future dates where
        # a stock may no longer pass today's liquidity filter.  Scores are
        # merged only where the stock was investable and the date passed the
        # cross-section coverage gate.  This avoids forward-return lookups
        # disappearing simply because future eligibility changed.
        panel_cols = [c for c in [
            'date','ticker','sector','exchange','open','high','low','close','volume','value',
            'value_avg_20','benchmark_close','ret_1','ret_5','ret_20','ret_60',
            'ma20','ma50','ma100','ma20_distance','ma50_distance','ma100_distance',
            'ma50_slope_20','ma100_slope_20','medium_trend_confirm',
            'weekly_close','weekly_ema10','weekly_ema30','weekly_ema10_slope4',
            'weekly_ret12','weekly_macd','weekly_macd_signal','weekly_macd_hist','weekly_trend_confirm',
            'macd','macd_signal','macd_hist','macd_positive','ma_bull','ma_cross_up','ma_cross_recent_10',
            'bb_upper','bb_lower','bb_bandwidth','bb_squeeze_recent_10','bb_breakout_after_squeeze',
            'prior_high_10','prior_high_20'
        ] if c in feat.columns]
        panel = feat[panel_cols].copy()
        panel['is_eligible'] = eligible.astype(bool).values
        panel['valid_cross_section'] = feat['date'].isin(valid_dates).values
        score_cols = ['date','ticker','leadership_score','short_momentum_score','long_momentum_score','acceleration','rs_score','flow_score','trend_score','sector_score','stage']
        panel = panel.merge(scored[score_cols], on=['date','ticker'], how='left')
        try:
            panel.to_parquet(out / 'research_panel.parquet', index=False)
        except Exception:
            panel.to_csv(out / 'research_panel.csv', index=False)

        sector_daily = (scored[['date','sector','sector_score','breadth_ma20','breadth_ma50']]
                        .drop_duplicates(['date','sector'])
                        .sort_values(['sector','date']))
        accw = int(model.get('acceleration_window', 5))
        sector_daily['acceleration'] = sector_daily.groupby('sector')['sector_score'].diff(accw)

        opp_cfg = cfg.get('opportunities', {})
        short_entries, long_entries = detect_opportunity_entries(
            scored,
            short_threshold=float(opp_cfg.get('short_threshold', 80)),
            long_threshold=float(opp_cfg.get('long_threshold', 80)),
            min_sector_score=float(opp_cfg.get('min_sector_score', 50)),
        )
        portfolio = build_model_portfolio(
            latest,
            size=int(opp_cfg.get('portfolio_size', 10)),
            sector_cap=int(opp_cfg.get('portfolio_sector_cap', 2)),
        )
        entry_signal_history = build_entry_signal_history(
            scored,
            min_sector_score=float(opp_cfg.get('entry_min_sector_score', 55)),
            min_leadership_score=float(opp_cfg.get('entry_min_leadership_score', 70)),
            min_short_score=float(opp_cfg.get('entry_min_short_score', 65)),
            min_long_score=float(opp_cfg.get('entry_min_long_score', 70)),
            require_macd_positive=bool(opp_cfg.get('entry_require_macd_positive', True)),
            require_medium_trend=bool(opp_cfg.get('entry_require_medium_trend', True)),
            require_weekly_trend=bool(opp_cfg.get('entry_require_weekly_trend', True)),
            max_ma20_distance=float(opp_cfg.get('entry_max_ma20_distance', 0.08)),
            max_ret5=float(opp_cfg.get('entry_max_ret5', 0.10)),
            squeeze_bonus=float(opp_cfg.get('entry_squeeze_bonus', 5)),
            pullback_bonus=float(opp_cfg.get('entry_pullback_bonus', 3)),
        )
        if not entry_signal_history.empty:
            entry_signal_history['model_version'] = str(
                opp_cfg.get('entry_model_version', 'v2-causal')
            )
        entry_candidates = build_entry_candidates(
            scored,
            top_n=int(opp_cfg.get('entry_top_n', 3)),
            min_sector_score=float(opp_cfg.get('entry_min_sector_score', 55)),
            min_leadership_score=float(opp_cfg.get('entry_min_leadership_score', 70)),
            min_short_score=float(opp_cfg.get('entry_min_short_score', 65)),
            min_long_score=float(opp_cfg.get('entry_min_long_score', 70)),
            require_macd_positive=bool(opp_cfg.get('entry_require_macd_positive', True)),
            require_medium_trend=bool(opp_cfg.get('entry_require_medium_trend', True)),
            require_weekly_trend=bool(opp_cfg.get('entry_require_weekly_trend', True)),
            max_ma20_distance=float(opp_cfg.get('entry_max_ma20_distance', 0.08)),
            max_ret5=float(opp_cfg.get('entry_max_ret5', 0.10)),
            max_age_sessions=int(opp_cfg.get('entry_max_age_sessions', 2)),
            max_distance_from_entry=float(opp_cfg.get('entry_max_distance_from_entry', 0.05)),
            squeeze_bonus=float(opp_cfg.get('entry_squeeze_bonus', 5)),
            pullback_bonus=float(opp_cfg.get('entry_pullback_bonus', 3)),
        )
        short_entries.to_csv(out / 'opportunities_short_latest.csv', index=False)
        long_entries.to_csv(out / 'opportunities_long_latest.csv', index=False)
        portfolio.to_csv(out / 'model_portfolio_10.csv', index=False)
        entry_candidates.to_csv(out / 'top3_entry_candidates.csv', index=False)
        entry_signal_history.to_csv(out / 'entry_signal_history.csv', index=False)

        regime.to_csv(out / 'market_regime.csv', index=False)
        render_dashboard(
            latest,
            reg_latest,
            out / 'dashboard.html',
            scored_history=scored,
            regime_history=regime,
            sector_history=sector_daily,
            opportunity_cfg=opp_cfg,
            price_history=feat,
        )
        docs = root / 'docs'
        docs.mkdir(exist_ok=True)
        render_dashboard(
            latest,
            reg_latest,
            docs / 'index.html',
            scored_history=scored,
            regime_history=regime,
            sector_history=sector_daily,
            opportunity_cfg=opp_cfg,
            price_history=feat,
        )

        db = DuckStore(root / 'data' / 'market_flow.duckdb')
        db.write_table('universe', universe)
        db.write_table('features_scored', scored)
        db.write_table('market_regime', regime)
        db.write_table('sector_daily', sector_daily)

        if sb is not None:
            sb.sync_universe(universe, as_of_date=latest_date)
            sb.sync_scores(latest)
            if not entry_signal_history.empty:
                sb.sync_entry_signals(entry_signal_history)
            sec_latest = sector_daily[sector_daily['date'] == latest_date].copy()
            sb.sync_sector(sec_latest)
            if not regime.empty:
                sb.sync_regime(regime[regime['date'] == latest_date].copy())
            sb.prune_completed_snapshot(
                latest_date,
                tickers=latest['ticker'].astype(str).tolist(),
                sectors=sec_latest['sector'].astype(str).tolist(),
            )
            sb.finish_run(run_id, provider_name, latest_date, len(latest), status='SUCCESS')

        return latest, regime
    except Exception as exc:
        if sb is not None:
            try:
                sb.finish_run(run_id, provider_name, None, 0, status='FAILED', message=str(exc)[:1000])
            except Exception:
                pass
        raise
