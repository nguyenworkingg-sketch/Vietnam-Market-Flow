from datetime import date

import pandas as pd
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

from marketflow.data.vnstock_provider import VNStockProvider
from marketflow.supabase_store import _clean_value


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


def test_universe_filters_non_stocks_and_uses_configured_icb_level():
    p = object.__new__(VNStockProvider)
    p.symbols = None
    p.sector_level = 2
    p._listing_obj = lambda: object()
    p._current_market_groups = lambda listing: pd.DataFrame({
        'ticker': ['AAA', 'BBB', 'CCC'],
        'exchange': ['HOSE', 'HNX', 'UPCOM'],
    })
    p._industry_frame = lambda listing: pd.DataFrame({
        'ticker': ['AAA', 'AAA', 'BBB', 'BBB', 'CCC', 'CCC', 'FUNDX', 'FUNDX'],
        'icb_level': [2, 4, 2, 4, 2, 4, 2, 4],
        'icb_name': ['Ngân hàng', 'Ngân hàng thương mại', 'Điện', 'Sản xuất điện', 'Công nghệ', 'Phần mềm', 'Tài chính', 'Quỹ'],
    })
    out = p.get_universe().set_index('ticker')
    # FUNDX exists in the broad industry/company source but is absent from the
    # current HOSE/HNX/UPCOM trading groups, so it must not be scanned.
    assert set(out.index) == {'AAA', 'BBB', 'CCC'}
    assert out.loc['AAA', 'sector'] == 'Ngân hàng'
    assert out.loc['BBB', 'sector'] == 'Điện'
    assert out.loc['CCC', 'sector'] == 'Công nghệ'


def test_clean_value_serializes_python_date():
    assert _clean_value(date(2026, 9, 18)) == '2026-09-18'
