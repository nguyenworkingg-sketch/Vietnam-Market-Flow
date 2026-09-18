from __future__ import annotations

from pathlib import Path
import pandas as pd


class CSVProvider:
    """Portable fallback provider.

    Expected files:
      prices.csv: date,ticker,open,high,low,close,volume[,value]
      universe.csv: ticker,exchange,sector[,name]
      benchmark.csv: date,open,high,low,close,volume
    """

    def __init__(self, folder: str | Path):
        self.folder = Path(folder)

    def get_universe(self) -> pd.DataFrame:
        df = pd.read_csv(self.folder/'universe.csv')
        df['ticker'] = df['ticker'].astype(str).str.upper()
        return df

    def get_prices(self, start: str, end: str | None = None) -> pd.DataFrame:
        df = pd.read_csv(self.folder/'prices.csv', parse_dates=['date'])
        mask = df['date'] >= pd.Timestamp(start)
        if end:
            mask &= df['date'] <= pd.Timestamp(end)
        df = df.loc[mask].copy()
        df['ticker'] = df['ticker'].astype(str).str.upper()
        if 'value' not in df.columns:
            df['value'] = df['close'] * df['volume']
        return df

    def get_benchmark(self, symbol: str, start: str, end: str | None = None) -> pd.DataFrame:
        df = pd.read_csv(self.folder/'benchmark.csv', parse_dates=['date'])
        mask = df['date'] >= pd.Timestamp(start)
        if end:
            mask &= df['date'] <= pd.Timestamp(end)
        return df.loc[mask].copy()
