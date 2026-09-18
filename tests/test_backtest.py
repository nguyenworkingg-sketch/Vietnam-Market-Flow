import numpy as np
import pandas as pd
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

from marketflow.backtest import run_research_suite


def _synthetic_panel():
    dates = pd.bdate_range('2025-01-02', periods=90)
    benchmark = 100 * (1.0005 ** np.arange(len(dates)))
    rows = []
    for i in range(20):
        drift = 0.0001 + i * 0.00012
        close = 20 * ((1 + drift) ** np.arange(len(dates)))
        for j, d in enumerate(dates):
            rows.append({
                'date': d,
                'ticker': f'T{i:02d}',
                'sector': 'A' if i < 10 else 'B',
                'close': close[j],
                'benchmark_close': benchmark[j],
                'is_eligible': True,
                'valid_cross_section': True,
                'leadership_score': 5 + i * 5,
                'acceleration': 0.0,
                'rs_score': 5 + i * 5,
                'flow_score': 50.0,
                'trend_score': 5 + i * 5,
                'sector_score': 50.0,
                'stage': 'LEADER' if i >= 16 else 'NEUTRAL',
            })
    return pd.DataFrame(rows)


def test_backtest_suite_produces_monotonic_signal_spread():
    suite = run_research_suite(_synthetic_panel(), horizons=(5, 20))
    summary = suite['summary'].set_index('horizon')
    assert summary.loc[5, 'top_bottom_spread'] > 0
    assert summary.loc[20, 'top_bottom_spread'] > 0
    assert summary.loc[5, 'mean_spearman_ic'] > 0
