from __future__ import annotations

import argparse
from datetime import timedelta
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from marketflow.utils import load_config, ensure_dirs
from marketflow.pipeline import run
from marketflow.data.csv_provider import CSVProvider
from marketflow.data.vnstock_provider import VNStockProvider


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--provider', choices=['vnstock', 'csv'], default='vnstock')
    ap.add_argument('--csv-folder', default='sample_data')
    ap.add_argument('--symbols', default='', help='Comma-separated smoke-test universe')
    ap.add_argument('--start', default='', help='Override history start YYYY-MM-DD')
    ap.add_argument('--full-history', action='store_true', help='Use config history_start instead of rolling live lookback')
    args = ap.parse_args()

    cfg = load_config(ROOT / 'config' / 'model.yaml')
    ensure_dirs(ROOT)

    symbols = [s.strip().upper() for s in args.symbols.split(',') if s.strip()] or None
    if symbols:
        # CI smoke tests intentionally use a tiny universe. Keep the same
        # coverage logic but scale its absolute floor to that test universe.
        cfg['model']['min_cross_section_stocks'] = max(2, int(len(symbols) * 0.70))

    if args.start:
        start = args.start
    elif args.full_history:
        start = cfg['model']['history_start']
    else:
        days = int(cfg['model'].get('live_lookback_days', 520))
        start = (pd.Timestamp.now(tz='Asia/Ho_Chi_Minh').date() - timedelta(days=days)).isoformat()

    if args.provider == 'vnstock':
        provider = VNStockProvider(
            symbols=symbols,
            live_max_symbols=None if symbols else int(cfg['model'].get('live_max_symbols', 450)),
            sector_level=int(cfg['model'].get('sector_level', 2)),
        )
    else:
        provider = CSVProvider(ROOT / args.csv_folder)

    latest, regime = run(provider, cfg, ROOT, history_start=start, provider_name=args.provider)
    print(f'Completed {len(latest)} eligible stocks. Latest date={latest.date.max()} start={start}')
    if not regime.empty:
        print(regime.tail(1).to_string(index=False))


if __name__ == '__main__':
    main()
