import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

from marketflow.data.vnstock_provider import VNStockProvider


def test_equity_price_normalization_from_kvnd():
    df = pd.DataFrame({'open':[28.0], 'high':[29.0], 'low':[27.5], 'close':[28.6], 'volume':[1_000_000]})
    out = VNStockProvider._normalize_equity_prices(df)
    assert out.loc[0, 'close'] == 28.6
    assert out.loc[0, 'value'] == 28_600_000_000


def test_equity_price_normalization_from_vnd():
    df = pd.DataFrame({'open':[28000], 'high':[29000], 'low':[27500], 'close':[28600], 'volume':[1_000_000]})
    out = VNStockProvider._normalize_equity_prices(df)
    assert out.loc[0, 'close'] == 28.6
    assert out.loc[0, 'value'] == 28_600_000_000
