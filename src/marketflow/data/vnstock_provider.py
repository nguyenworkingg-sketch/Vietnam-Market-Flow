from __future__ import annotations

import os
import time
from typing import Iterable

import numpy as np
import pandas as pd


class VNStockProvider:
    """Vnstock v4+ adapter.

    The quantitative model only consumes a small canonical schema, so provider-
    specific quirks stay isolated here. Equity prices are normalized to
    *thousand VND per share* to keep thresholds consistent across sources.
    Trading value is normalized to VND.
    """

    def __init__(
        self,
        source: str = "KBS",
        listing_source: str = "VCI",
        sleep: float = 0.05,
        symbols: Iterable[str] | None = None,
        live_max_symbols: int | None = None,
    ):
        self.source = source
        self.listing_source = listing_source
        self.sleep = sleep
        self.symbols = [str(s).upper() for s in symbols] if symbols else None
        self.live_max_symbols = live_max_symbols

        try:
            from vnstock import Listing, Market, register_user
        except Exception as e:
            raise ImportError("Install vnstock>=4.0.8 or use CSVProvider.") from e

        self.Listing = Listing
        self.Market = Market

        api_key = os.getenv("VNSTOCK_API_KEY", "").strip()
        if api_key:
            try:
                register_user(api_key=api_key)
            except Exception as exc:
                print(f"[WARN] Could not register VNSTOCK_API_KEY: {exc}")

    @staticmethod
    def _normalize_equity_prices(df: pd.DataFrame) -> pd.DataFrame:
        x = df.copy()
        price_cols = [c for c in ["open", "high", "low", "close"] if c in x.columns]
        if not price_cols:
            return x
        med = pd.to_numeric(x["close"], errors="coerce").median()
        if pd.notna(med) and med > 1000:
            x[price_cols] = x[price_cols].apply(pd.to_numeric, errors="coerce") / 1000.0
        else:
            x[price_cols] = x[price_cols].apply(pd.to_numeric, errors="coerce")
        x["volume"] = pd.to_numeric(x["volume"], errors="coerce")
        if "value" not in x.columns:
            x["value"] = x["close"] * 1000.0 * x["volume"]
        else:
            x["value"] = pd.to_numeric(x["value"], errors="coerce")
            implied = x["close"] * 1000.0 * x["volume"]
            ratio = (x["value"] / implied.replace(0, np.nan)).median()
            if pd.notna(ratio) and ratio < 0.1:
                x["value"] = x["value"] * 1000.0
        return x

    def _listing_obj(self):
        for source in [self.listing_source, "KBS", None]:
            try:
                return self.Listing(source=source) if source else self.Listing()
            except Exception:
                continue
        return self.Listing()

    @staticmethod
    def _exchange_frame(raw) -> pd.DataFrame:
        if isinstance(raw, pd.DataFrame):
            x = raw.copy()
            if "symbol" in x.columns:
                x = x.rename(columns={"symbol": "ticker"})
            if "ticker" not in x.columns:
                raise ValueError("symbols_by_exchange DataFrame has no symbol/ticker column")
            return x
        if isinstance(raw, dict):
            rows = []
            for exchange, syms in raw.items():
                for sym in syms or []:
                    rows.append({"ticker": str(sym), "exchange": str(exchange).upper()})
            return pd.DataFrame(rows)
        if isinstance(raw, (list, tuple, set)):
            return pd.DataFrame({"ticker": list(raw)})
        raise TypeError(f"Unsupported symbols_by_exchange result: {type(raw)!r}")

    def _all_exchange_symbols(self, listing) -> pd.DataFrame:
        try:
            raw = listing.symbols_by_exchange()
            x = self._exchange_frame(raw)
            if "exchange" in x.columns and not x.empty:
                return x
        except Exception:
            pass

        frames = []
        for exchange in ["HOSE", "HNX", "UPCOM"]:
            try:
                raw = listing.symbols_by_exchange(exchange=exchange)
            except TypeError:
                raw = listing.symbols_by_exchange(exchange)
            x = self._exchange_frame(raw)
            if "exchange" not in x.columns:
                x["exchange"] = exchange
            frames.append(x)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["ticker", "exchange"])

    def _industry_frame(self, listing) -> pd.DataFrame:
        for lang in ["vi", "en", None]:
            try:
                raw = listing.symbols_by_industries(lang=lang) if lang else listing.symbols_by_industries()
                if isinstance(raw, pd.DataFrame):
                    x = raw.copy()
                    if "symbol" in x.columns:
                        x = x.rename(columns={"symbol": "ticker"})
                    return x
            except Exception:
                continue
        return pd.DataFrame(columns=["ticker", "sector"])

    def get_universe(self) -> pd.DataFrame:
        listing = self._listing_obj()
        exch = self._all_exchange_symbols(listing)
        if exch.empty and self.symbols:
            # Smoke/debug mode must not depend on the listing endpoint being
            # available. Production still requires a valid listing universe.
            exch = pd.DataFrame({"ticker": self.symbols, "exchange": "UNKNOWN"})
        exch["ticker"] = exch["ticker"].astype(str).str.upper()
        if "exchange" in exch.columns:
            exch["exchange"] = exch["exchange"].astype(str).str.upper()
            if not self.symbols:
                exch = exch[exch["exchange"].isin(["HOSE", "HNX", "UPCOM"])].copy()

        ind = self._industry_frame(listing)
        if not ind.empty:
            ind["ticker"] = ind["ticker"].astype(str).str.upper()
            if "icb_level" in ind.columns:
                ind = ind.sort_values(["ticker", "icb_level"]).groupby("ticker").tail(1)
            sector_col = next((c for c in ["icb_name", "industry", "industry_name", "sector"] if c in ind.columns), None)
            if sector_col:
                ind = ind[["ticker", sector_col]].rename(columns={sector_col: "sector"}).drop_duplicates("ticker")
                exch = exch.merge(ind, on="ticker", how="left")
        if "sector" not in exch.columns:
            exch["sector"] = "Chưa phân ngành"
        exch["sector"] = exch["sector"].fillna("Chưa phân ngành")

        if self.symbols:
            exch = exch[exch["ticker"].isin(self.symbols)].copy()
            present = set(exch["ticker"].astype(str))
            missing = [s for s in self.symbols if s not in present]
            if missing:
                exch = pd.concat(
                    [exch, pd.DataFrame({"ticker": missing, "exchange": "UNKNOWN", "sector": "Chưa phân ngành"})],
                    ignore_index=True,
                )
        if "organ_name" in exch.columns and "name" not in exch.columns:
            exch = exch.rename(columns={"organ_name": "name"})
        return exch.drop_duplicates("ticker").reset_index(drop=True)

    def _ohlcv(self, kind: str, symbol: str, start: str, end: str | None) -> pd.DataFrame:
        market = self.Market()
        domain = getattr(market, kind)
        kwargs = {"symbol": symbol, "start": start}
        if end:
            kwargs["end"] = end
        try:
            df = domain.ohlcv(**kwargs)
        except (TypeError, AttributeError):
            obj = domain(symbol) if callable(domain) else domain
            kwargs2 = {"start": start}
            if end:
                kwargs2["end"] = end
            try:
                df = obj.ohlcv(**kwargs2)
            except TypeError:
                kwargs3 = {"start_date": start}
                if end:
                    kwargs3["end_date"] = end
                df = obj.ohlcv(**kwargs3)
        df = pd.DataFrame(df).copy()
        if "time" in df.columns:
            df = df.rename(columns={"time": "date"})
        if "date" not in df.columns:
            raise ValueError(f"No date/time column returned for {symbol}")
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        return df

    def _price_board(self, symbols: list[str]) -> pd.DataFrame:
        if not symbols:
            return pd.DataFrame()
        market = self.Market()
        try:
            return pd.DataFrame(market.price_board(symbols))
        except Exception:
            try:
                from vnstock import Trading
                return pd.DataFrame(Trading(source=self.source, symbol=symbols[0]).price_board(symbols_list=symbols))
            except Exception as exc:
                print(f"[WARN] Price-board prefilter unavailable: {exc}")
                return pd.DataFrame()

    def _select_live_symbols(self, universe: pd.DataFrame) -> list[str]:
        syms = universe["ticker"].astype(str).tolist()
        if self.live_max_symbols is None or len(syms) <= self.live_max_symbols:
            return syms
        boards = []
        for i in range(0, len(syms), 100):
            batch = syms[i:i+100]
            b = self._price_board(batch)
            if not b.empty:
                boards.append(b)
            time.sleep(self.sleep)
        if not boards:
            return syms[: self.live_max_symbols]
        board = pd.concat(boards, ignore_index=True)
        symcol = next((c for c in ["symbol", "ticker"] if c in board.columns), None)
        valcol = next((c for c in ["total_value", "value", "trading_value"] if c in board.columns), None)
        if not symcol or not valcol:
            return syms[: self.live_max_symbols]
        board[valcol] = pd.to_numeric(board[valcol], errors="coerce").fillna(0)
        selected = board.sort_values(valcol, ascending=False)[symcol].astype(str).str.upper().head(self.live_max_symbols).tolist()
        return selected

    def get_prices_for_symbols(self, symbols: Iterable[str], start: str, end: str | None = None) -> pd.DataFrame:
        frames = []
        for sym in [str(s).upper() for s in symbols]:
            try:
                d = self._ohlcv("equity", sym, start, end)
                d = self._normalize_equity_prices(d)
                d["ticker"] = sym
                frames.append(d)
            except Exception as exc:
                print(f"[WARN] {sym}: {exc}")
            time.sleep(self.sleep)
        if not frames:
            raise RuntimeError("No equity data returned from provider.")
        return pd.concat(frames, ignore_index=True)

    def get_prices(self, start: str, end: str | None = None) -> pd.DataFrame:
        if self.symbols:
            symbols = self.symbols
        else:
            universe = self.get_universe()
            symbols = self._select_live_symbols(universe)
        return self.get_prices_for_symbols(symbols, start, end)

    def get_benchmark(self, symbol: str, start: str, end: str | None = None) -> pd.DataFrame:
        try:
            d = self._ohlcv("index", symbol, start, end)
        except Exception:
            d = self._ohlcv("equity", symbol, start, end)
        d["volume"] = pd.to_numeric(d.get("volume"), errors="coerce")
        d["close"] = pd.to_numeric(d.get("close"), errors="coerce")
        if "value" not in d.columns:
            d["value"] = d["close"] * d["volume"]
        return d
