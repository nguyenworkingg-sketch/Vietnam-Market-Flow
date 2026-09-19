from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from datetime import date, datetime
from urllib import request, error
from urllib.parse import urlencode

import numpy as np
import pandas as pd


def _clean_value(v):
    if v is None:
        return None
    if isinstance(v, (pd.Timestamp, datetime, date)):
        return v.isoformat()
    if isinstance(v, np.datetime64):
        return pd.Timestamp(v).isoformat()
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, (np.floating, float)):
        x = float(v)
        return None if math.isnan(x) or math.isinf(x) else x
    if pd.isna(v):
        return None
    return v


def _records(df: pd.DataFrame, rename: dict[str, str] | None = None) -> list[dict]:
    x = df.copy()
    if rename:
        x = x.rename(columns=rename)
    if 'trade_date' in x.columns:
        x['trade_date'] = pd.to_datetime(x['trade_date']).dt.date
    return [{k: _clean_value(v) for k, v in row.items()} for row in x.to_dict('records')]


@dataclass
class SupabaseRESTStore:
    url: str
    key: str
    batch_size: int = 500

    @classmethod
    def from_env(cls) -> "SupabaseRESTStore | None":
        url = os.getenv('SUPABASE_URL', '').strip().rstrip('/')
        key = os.getenv('SUPABASE_SERVICE_ROLE_KEY', '').strip()
        if not url or not key:
            return None
        return cls(url=url, key=key)

    def _headers(self, prefer: str | None = None) -> dict[str, str]:
        h = {
            'apikey': self.key,
            'Authorization': f'Bearer {self.key}',
            'Content-Type': 'application/json',
        }
        if prefer:
            h['Prefer'] = prefer
        return h

    def _post(
        self,
        table: str,
        rows: list[dict],
        on_conflict: str | None = None,
        return_representation: bool = False,
    ):
        if not rows:
            return []
        endpoint = f'{self.url}/rest/v1/{table}'
        if on_conflict:
            endpoint += f'?on_conflict={on_conflict}'
        payload = json.dumps(rows, ensure_ascii=False).encode('utf-8')
        ret = 'representation' if return_representation else 'minimal'
        req = request.Request(
            endpoint,
            data=payload,
            headers=self._headers(f'resolution=merge-duplicates,return={ret}'),
            method='POST',
        )
        try:
            with request.urlopen(req, timeout=60) as r:
                body = r.read().decode('utf-8')
                return json.loads(body) if body else []
        except error.HTTPError as e:
            body = e.read().decode('utf-8', errors='replace')
            raise RuntimeError(f'Supabase write failed table={table} status={e.code}: {body}') from e

    def _patch(self, table: str, query: str, values: dict) -> None:
        endpoint = f'{self.url}/rest/v1/{table}?{query}'
        payload = json.dumps({k: _clean_value(v) for k, v in values.items()}, ensure_ascii=False).encode('utf-8')
        req = request.Request(
            endpoint,
            data=payload,
            headers=self._headers('return=minimal'),
            method='PATCH',
        )
        try:
            with request.urlopen(req, timeout=60) as r:
                r.read()
        except error.HTTPError as e:
            body = e.read().decode('utf-8', errors='replace')
            raise RuntimeError(f'Supabase patch failed table={table} status={e.code}: {body}') from e

    def _delete_where(self, table: str, params: dict[str, str]) -> None:
        endpoint = f'{self.url}/rest/v1/{table}?' + urlencode(params, safe='(),.*')
        req = request.Request(
            endpoint,
            headers=self._headers('return=minimal'),
            method='DELETE',
        )
        try:
            with request.urlopen(req, timeout=60) as r:
                r.read()
        except error.HTTPError as e:
            body = e.read().decode('utf-8', errors='replace')
            raise RuntimeError(f'Supabase delete failed table={table} status={e.code}: {body}') from e

    def _get_rows(self, table: str, params: dict[str, str], page_size: int = 1000) -> list[dict]:
        rows: list[dict] = []
        offset = 0
        while True:
            endpoint = f'{self.url}/rest/v1/{table}?' + urlencode(params, safe='(),.*')
            headers = self._headers()
            headers['Range-Unit'] = 'items'
            headers['Range'] = f'{offset}-{offset + page_size - 1}'
            req = request.Request(endpoint, headers=headers, method='GET')
            try:
                with request.urlopen(req, timeout=60) as r:
                    body = r.read().decode('utf-8')
            except error.HTTPError as e:
                body = e.read().decode('utf-8', errors='replace')
                raise RuntimeError(f'Supabase read failed table={table} status={e.code}: {body}') from e
            batch = json.loads(body) if body else []
            rows.extend(batch)
            if len(batch) < page_size:
                break
            offset += page_size
        return rows

    def _batch_post(self, table: str, rows: list[dict], on_conflict: str | None = None):
        for i in range(0, len(rows), self.batch_size):
            self._post(table, rows[i:i+self.batch_size], on_conflict=on_conflict)

    def start_run(self, provider: str) -> int | None:
        rows = self._post(
            'mf_runs',
            [{'status': 'RUNNING', 'provider': provider}],
            return_representation=True,
        )
        if rows and rows[0].get('run_id') is not None:
            return int(rows[0]['run_id'])
        return None

    def finish_run(
        self,
        run_id: int | None,
        provider: str,
        latest_trade_date,
        eligible_stocks: int,
        status: str = 'SUCCESS',
        message: str | None = None,
    ) -> None:
        values = {
            'status': status,
            'provider': provider,
            'latest_trade_date': pd.Timestamp(latest_trade_date).date() if latest_trade_date is not None else None,
            'eligible_stocks': int(eligible_stocks),
            'message': message,
            'finished_at': pd.Timestamp.now(tz='UTC'),
        }
        if run_id is None:
            self._post('mf_runs', [{k: _clean_value(v) for k, v in values.items()}])
        else:
            self._patch('mf_runs', f'run_id=eq.{int(run_id)}', values)

    def fetch_ohlcv(self, start_date, tickers: list[str] | None = None) -> pd.DataFrame:
        params = {
            'select': 'trade_date,ticker,open,high,low,close,volume,value',
            'trade_date': f'gte.{pd.Timestamp(start_date).date().isoformat()}',
            'order': 'trade_date.asc,ticker.asc',
        }
        if tickers:
            params['ticker'] = 'in.(' + ','.join(sorted(set(map(str, tickers)))) + ')'
        rows = self._get_rows('mf_ohlcv_daily', params)
        if not rows:
            return pd.DataFrame(columns=['date','ticker','open','high','low','close','volume','value'])
        x = pd.DataFrame(rows).rename(columns={'trade_date':'date'})
        x['date'] = pd.to_datetime(x['date']).dt.normalize()
        for col in ['open','high','low','close','volume','value']:
            if col in x.columns:
                x[col] = pd.to_numeric(x[col], errors='coerce')
        return x

    def sync_ohlcv(self, prices: pd.DataFrame) -> None:
        cols = ['date','ticker','open','high','low','close','volume','value']
        x = prices[[col for col in cols if col in prices.columns]].copy()
        x = x.dropna(subset=['date','ticker','close']).drop_duplicates(['date','ticker'], keep='last')
        rows = _records(x, rename={'date':'trade_date'})
        self._batch_post('mf_ohlcv_daily', rows, on_conflict='trade_date,ticker')

    def sync_universe(self, universe: pd.DataFrame, as_of_date=None) -> None:
        cols = [c for c in ['ticker', 'exchange', 'sector', 'name'] if c in universe.columns]
        x = universe[cols].copy()
        if 'name' not in x.columns and 'organ_name' in universe.columns:
            x['name'] = universe['organ_name']
        seen = pd.Timestamp(as_of_date).date() if as_of_date is not None else pd.Timestamp.now(tz='Asia/Ho_Chi_Minh').date()
        # Current production universe is a snapshot, so explicitly deactivate
        # prior rows first; this prevents old ETFs/derivatives from remaining
        # "active" after universe rules change.
        self._patch('mf_universe', 'is_active=eq.true', {'is_active': False})
        x['is_active'] = True
        x['last_seen'] = seen
        rows = _records(x.drop_duplicates('ticker'))
        self._batch_post('mf_universe', rows, on_conflict='ticker')

    def sync_scores(self, scored: pd.DataFrame) -> None:
        cols = ['date','ticker','sector','rs_score','flow_score','trend_score','sector_score','leadership_score','short_momentum_score','long_momentum_score','acceleration','stage']
        x = scored[[c for c in cols if c in scored.columns]].copy()
        rows = _records(x, rename={'date':'trade_date'})
        self._batch_post('mf_scores_daily', rows, on_conflict='trade_date,ticker')

    def sync_sector(self, sector_daily: pd.DataFrame) -> None:
        cols = ['date','sector','sector_score','acceleration','breadth_ma20','breadth_ma50']
        x = sector_daily[[c for c in cols if c in sector_daily.columns]].copy()
        rows = _records(x, rename={'date':'trade_date'})
        self._batch_post('mf_sector_daily', rows, on_conflict='trade_date,sector')

    def prune_completed_snapshot(self, latest_trade_date, tickers: list[str], sectors: list[str]) -> None:
        dt = pd.Timestamp(latest_trade_date).date().isoformat()

        # Any later date is, by definition, an incomplete snapshot if the
        # coverage-gated engine selected an earlier authoritative as-of date.
        for table in ['mf_scores_daily', 'mf_sector_daily', 'mf_market_regime']:
            self._delete_where(table, {'trade_date': f'gt.{dt}'})

        if tickers:
            ticker_filter = 'not.in.(' + ','.join(sorted(set(map(str, tickers)))) + ')'
            self._delete_where(
                'mf_scores_daily',
                {'trade_date': f'eq.{dt}', 'ticker': ticker_filter},
            )
        if sectors:
            # Quote individual sector strings in the PostgREST in-list grammar.
            safe_sectors = [str(s).replace('"', '') for s in sorted(set(sectors))]
            sector_filter = 'not.in.(' + ','.join(f'"{s}"' for s in safe_sectors) + ')'
            self._delete_where(
                'mf_sector_daily',
                {'trade_date': f'eq.{dt}', 'sector': sector_filter},
            )

    def sync_regime(self, regime: pd.DataFrame) -> None:
        cols = ['date','market_score','breadth_ma20','breadth_ma50','median_ret20','liquidity_ratio','regime','coverage_stocks','coverage_reference']
        x = regime[[c for c in cols if c in regime.columns]].copy()
        rows = _records(x, rename={'date':'trade_date'})
        self._batch_post('mf_market_regime', rows, on_conflict='trade_date')
