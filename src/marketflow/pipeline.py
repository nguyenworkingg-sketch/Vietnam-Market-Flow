from __future__ import annotations

import math
from pathlib import Path
import pandas as pd

from .features import add_stock_features, add_sector_features
from .scoring import build_scores, add_stage
from .regime import market_regime
from .storage import DuckStore
from .dashboard import render_dashboard
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
        prices = provider.get_prices(start)
        bench = provider.get_benchmark(model['benchmark'], start)

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
        hist = scored[['date','ticker','sector','leadership_score','acceleration','rs_score','flow_score','trend_score','sector_score','stage']]
        try:
            hist.to_parquet(out / 'scores_history.parquet', index=False)
        except Exception:
            hist.to_csv(out / 'scores_history.csv', index=False)
        regime.to_csv(out / 'market_regime.csv', index=False)
        render_dashboard(latest, reg_latest, out / 'dashboard.html')
        docs = root / 'docs'
        docs.mkdir(exist_ok=True)
        render_dashboard(latest, reg_latest, docs / 'index.html')

        db = DuckStore(root / 'data' / 'market_flow.duckdb')
        db.write_table('universe', universe)
        db.write_table('features_scored', scored)
        db.write_table('market_regime', regime)

        sector_daily = (scored[['date','sector','sector_score','breadth_ma20','breadth_ma50']]
                        .drop_duplicates(['date','sector'])
                        .sort_values(['sector','date']))
        accw = int(model.get('acceleration_window', 5))
        sector_daily['acceleration'] = sector_daily.groupby('sector')['sector_score'].diff(accw)
        db.write_table('sector_daily', sector_daily)

        if sb is not None:
            sb.sync_universe(universe)
            sb.sync_scores(latest)
            sec_latest = sector_daily[sector_daily['date'] == latest_date].copy()
            sb.sync_sector(sec_latest)
            if not regime.empty:
                sb.sync_regime(regime[regime['date'] == latest_date].copy())
            sb.finish_run(run_id, provider_name, latest_date, len(latest), status='SUCCESS')

        return latest, regime
    except Exception as exc:
        if sb is not None:
            try:
                sb.finish_run(run_id, provider_name, None, 0, status='FAILED', message=str(exc)[:1000])
            except Exception:
                pass
        raise
