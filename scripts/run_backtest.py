from __future__ import annotations

from pathlib import Path
import sys
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from marketflow.backtest import decile_study


def main():
    p=ROOT/'outputs'/'scores_history.parquet'
    if p.exists():
        scores=pd.read_parquet(p)
    else:
        scores=pd.read_csv(ROOT/'outputs'/'scores_history.csv', parse_dates=['date'])
    print('Score-history rows:', len(scores))
    print('For production backtest, merge scores with raw close prices then call decile_study().')

if __name__=='__main__':
    main()
