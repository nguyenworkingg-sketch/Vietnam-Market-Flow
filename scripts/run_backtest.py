from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

from marketflow.backtest import run_research_suite


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
    for name in ['deciles', 'stage_entries', 'ic_daily', 'ic_summary', 'summary']:
        suite[name].to_csv(out/f'{name}.csv', index=False)

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
