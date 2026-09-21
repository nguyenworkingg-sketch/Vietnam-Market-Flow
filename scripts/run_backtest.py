from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

from marketflow.backtest import run_research_suite, entry_signal_study
from marketflow.dashboard import render_dashboard


def _load_panel() -> pd.DataFrame:
    pq = ROOT/'outputs'/'research_panel.parquet'
    csv = ROOT/'outputs'/'research_panel.csv'
    if pq.exists():
        return pd.read_parquet(pq)
    if csv.exists():
        return pd.read_csv(csv, parse_dates=['date'])
    raise FileNotFoundError(
        'research_panel not found. Run the daily pipeline first so price paths '
        'and model scores are generated together.'
    )


def main():
    panel = _load_panel()
    suite = run_research_suite(panel, horizons=(5, 20, 60))

    out = ROOT/'outputs'/'backtest'
    out.mkdir(parents=True, exist_ok=True)
    for name in ['deciles', 'stage_entries', 'ic_daily', 'ic_summary', 'summary', 'score_comparison']:
        suite[name].to_csv(out/f'{name}.csv', index=False)

    entry_path = ROOT/'outputs'/'entry_signal_history.csv'
    entry_audit_path = ROOT/'outputs'/'entry_signal_audit.csv'
    entry_audit = pd.DataFrame()
    if entry_audit_path.exists():
        entry_audit = pd.read_csv(entry_audit_path, parse_dates=['entry_date'])
    entry_summary = pd.DataFrame()
    if entry_path.exists():
        entries = pd.read_csv(entry_path, parse_dates=['entry_date'])
        entry_summary = entry_signal_study(suite['prepared'], entries, horizons=(5,20,60))
        entry_summary.to_csv(out/'entry_signal_summary.csv', index=False)

    # Re-render the final published dashboard after research diagnostics exist,
    # so the web page includes validation charts from the same production run.
    latest = pd.read_csv(ROOT/'outputs'/'scores_latest.csv', parse_dates=['date'])
    regime = pd.read_csv(ROOT/'outputs'/'market_regime.csv', parse_dates=['date'])
    latest_date = pd.to_datetime(latest['date']).max()
    reg_latest_df = regime[pd.to_datetime(regime['date']).eq(latest_date)]
    reg_latest = reg_latest_df.iloc[-1].to_dict() if not reg_latest_df.empty else {}

    scored = panel[panel['leadership_score'].notna()].copy()
    scored['date'] = pd.to_datetime(scored['date'])
    sector_history = (scored[['date','sector','sector_score']]
                      .dropna(subset=['sector','sector_score'])
                      .groupby(['date','sector'], as_index=False)['sector_score']
                      .median()
                      .sort_values(['sector','date']))
    sector_history['acceleration'] = sector_history.groupby('sector')['sector_score'].diff(5)

    bt_payload = {
        'summary': suite['summary'],
        'deciles': suite['deciles'],
        'ic_daily': suite['ic_daily'],
        'stage_entries': suite['stage_entries'],
        'ic_summary': suite['ic_summary'],
        'entry_signal_summary': entry_summary,
        'score_comparison': suite['score_comparison'],
    }
    cfg = yaml.safe_load((ROOT/'config'/'model.yaml').read_text(encoding='utf-8'))
    opp_cfg = cfg.get('opportunities', {})
    risk_cfg = cfg.get('risk_management', {})
    render_dashboard(
        latest, reg_latest, ROOT/'outputs'/'dashboard.html',
        scored_history=scored,
        regime_history=regime,
        sector_history=sector_history,
        backtest=bt_payload,
        opportunity_cfg=opp_cfg,
        risk_cfg=risk_cfg,
        price_history=panel,
        historical_entry_events=entry_audit,
    )
    (ROOT/'docs').mkdir(exist_ok=True)
    render_dashboard(
        latest, reg_latest, ROOT/'docs'/'index.html',
        scored_history=scored,
        regime_history=regime,
        sector_history=sector_history,
        backtest=bt_payload,
        opportunity_cfg=opp_cfg,
        risk_cfg=risk_cfg,
        price_history=panel,
        historical_entry_events=entry_audit,
    )

    signal_rows = int(panel['leadership_score'].notna().sum())
    dates = pd.to_datetime(panel.loc[panel['leadership_score'].notna(), 'date'])
    print(f'Research panel rows: {len(panel):,}')
    print(f'Signal observations: {signal_rows:,}')
    if len(dates):
        print(f'Signal period: {dates.min().date()} -> {dates.max().date()}')
    print('\nPRELIMINARY BACKTEST SUMMARY')
    if suite['summary'].empty:
        print('Not enough forward observations yet.')
    else:
        display = suite['summary'].copy()
        pct_cols = [c for c in display.columns if 'alpha' in c or 'spread' in c or 'rate' in c]
        for col in pct_cols:
            display[col] = display[col].map(lambda v: '' if pd.isna(v) else f'{100*v:.2f}%')
        print(display.to_string(index=False))

    print(
        '\nCAVEAT: this live research panel is selected from the current production '
        'universe. Treat results as diagnostic, not survivorship-bias-free proof, '
        'until the historical universe/delisted-symbol backfill is completed.'
    )


if __name__ == '__main__':
    main()
