from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


def add_forward_returns(panel: pd.DataFrame, horizons: Iterable[int] = (5, 20, 60)) -> pd.DataFrame:
    """Add calendar-aligned forward returns and market/sector alpha.

    Horizons are measured in benchmark trading sessions, not in observations of
    each individual stock.  If a stock has no close on the target session
    (e.g. suspension), the forward return is left missing rather than silently
    shifting to another date.
    """
    x = panel.copy()
    x['date'] = pd.to_datetime(x['date']).dt.normalize()
    x = x.sort_values(['ticker', 'date']).reset_index(drop=True)

    cal = (x[['date', 'benchmark_close']]
           .dropna(subset=['benchmark_close'])
           .drop_duplicates('date')
           .sort_values('date')
           .reset_index(drop=True))

    base_prices = x[['ticker', 'date', 'close']].drop_duplicates(['ticker', 'date'])

    for h in horizons:
        target_col = f'target_date_{h}'
        mkt_col = f'market_fwd_{h}'
        cal_h = cal[['date', 'benchmark_close']].copy()
        cal_h[target_col] = cal_h['date'].shift(-h)
        cal_h[mkt_col] = cal_h['benchmark_close'].shift(-h) / cal_h['benchmark_close'] - 1
        x = x.merge(cal_h[['date', target_col, mkt_col]], on='date', how='left')

        future = base_prices.rename(columns={'date': target_col, 'close': f'future_close_{h}'})
        x = x.merge(future, on=['ticker', target_col], how='left')
        x[f'fwd_{h}'] = x[f'future_close_{h}'] / x['close'] - 1
        x[f'alpha_market_{h}'] = x[f'fwd_{h}'] - x[mkt_col]

        # Equal-weight median sector return among names that were investable on
        # the signal date. Median is intentionally robust to penny outliers.
        base_mask = (
            x.get('is_eligible', True).astype(bool)
            & x.get('valid_cross_section', True).astype(bool)
        )
        sec = (x.loc[base_mask, ['date', 'sector', f'fwd_{h}']]
               .dropna(subset=[f'fwd_{h}'])
               .groupby(['date', 'sector'], as_index=False)[f'fwd_{h}']
               .median()
               .rename(columns={f'fwd_{h}': f'sector_fwd_{h}'}))
        x = x.merge(sec, on=['date', 'sector'], how='left')
        x[f'alpha_sector_{h}'] = x[f'fwd_{h}'] - x[f'sector_fwd_{h}']

    return x


def _decile(s: pd.Series) -> pd.Series:
    valid = s.notna()
    out = pd.Series(np.nan, index=s.index)
    if valid.sum() < 10:
        return out
    ranks = s.loc[valid].rank(method='first')
    out.loc[valid] = pd.qcut(ranks, 10, labels=False, duplicates='drop') + 1
    return out


def decile_study(prepared: pd.DataFrame, horizons: Iterable[int] = (5, 20, 60)) -> pd.DataFrame:
    x = prepared[prepared['leadership_score'].notna()].copy()
    x['score_decile'] = x.groupby('date')['leadership_score'].transform(_decile)

    rows = []
    metrics = {
        'absolute': 'fwd_{h}',
        'market_alpha': 'alpha_market_{h}',
        'sector_alpha': 'alpha_sector_{h}',
    }
    for h in horizons:
        for label, pattern in metrics.items():
            col = pattern.format(h=h)
            d = x.dropna(subset=['score_decile', col]).copy()
            for decile, g in d.groupby('score_decile'):
                rows.append({
                    'horizon': h,
                    'metric': label,
                    'decile': int(decile),
                    'mean': g[col].mean(),
                    'median': g[col].median(),
                    'hit_rate': (g[col] > 0).mean(),
                    'count': len(g),
                })
    return pd.DataFrame(rows, columns=['horizon','metric','decile','mean','median','hit_rate','count'])


def stage_entry_study(prepared: pd.DataFrame, horizons: Iterable[int] = (5, 20, 60)) -> pd.DataFrame:
    x = prepared[prepared['leadership_score'].notna()].copy().sort_values(['ticker', 'date'])
    x['prev_stage'] = x.groupby('ticker')['stage'].shift(1)
    x = x[x['prev_stage'].notna() & x['stage'].ne(x['prev_stage'])].copy()

    rows = []
    for h in horizons:
        for metric, col in [
            ('absolute', f'fwd_{h}'),
            ('market_alpha', f'alpha_market_{h}'),
            ('sector_alpha', f'alpha_sector_{h}'),
        ]:
            d = x.dropna(subset=[col])
            for stage, g in d.groupby('stage'):
                rows.append({
                    'horizon': h,
                    'metric': metric,
                    'stage': stage,
                    'mean': g[col].mean(),
                    'median': g[col].median(),
                    'hit_rate': (g[col] > 0).mean(),
                    'count': len(g),
                })
    return pd.DataFrame(rows, columns=['horizon','metric','stage','mean','median','hit_rate','count'])


def information_coefficient(prepared: pd.DataFrame, horizons: Iterable[int] = (5, 20, 60)) -> tuple[pd.DataFrame, pd.DataFrame]:
    x = prepared[prepared['leadership_score'].notna()].copy()
    daily_rows = []
    for h in horizons:
        for metric, col in [
            ('market_alpha', f'alpha_market_{h}'),
            ('sector_alpha', f'alpha_sector_{h}'),
        ]:
            for dt, g in x[['date', 'leadership_score', col]].dropna().groupby('date'):
                if len(g) < 20:
                    continue
                ic = g['leadership_score'].corr(g[col], method='spearman')
                daily_rows.append({'date': dt, 'horizon': h, 'metric': metric, 'ic': ic, 'n': len(g)})
    daily = pd.DataFrame(daily_rows)
    if daily.empty:
        return (
            pd.DataFrame(columns=['date','horizon','metric','ic','n']),
            pd.DataFrame(columns=['horizon','metric','mean_ic','median_ic','ic_positive_rate','ic_std','days','ic_ir']),
        )
    summary = (daily.groupby(['horizon', 'metric'], as_index=False)
               .agg(mean_ic=('ic', 'mean'),
                    median_ic=('ic', 'median'),
                    ic_positive_rate=('ic', lambda s: (s > 0).mean()),
                    ic_std=('ic', 'std'),
                    days=('ic', 'count')))
    summary['ic_ir'] = summary['mean_ic'] / summary['ic_std'].replace(0, np.nan)
    return daily, summary


def research_summary(
    deciles: pd.DataFrame,
    stages: pd.DataFrame,
    ic_summary: pd.DataFrame,
    horizons: Iterable[int] = (5, 20, 60),
) -> pd.DataFrame:
    rows = []
    for h in horizons:
        d = deciles[(deciles['horizon'] == h) & (deciles['metric'] == 'market_alpha')]
        top = d[d['decile'] == d['decile'].max()] if not d.empty else pd.DataFrame()
        bottom = d[d['decile'] == d['decile'].min()] if not d.empty else pd.DataFrame()
        ic = ic_summary[(ic_summary['horizon'] == h) & (ic_summary['metric'] == 'market_alpha')]
        emerging = stages[(stages['horizon'] == h) & (stages['metric'] == 'market_alpha') & (stages['stage'] == 'EMERGING')]
        rows.append({
            'horizon': h,
            'top_decile_market_alpha': top['mean'].iloc[0] if len(top) else np.nan,
            'bottom_decile_market_alpha': bottom['mean'].iloc[0] if len(bottom) else np.nan,
            'top_bottom_spread': (
                top['mean'].iloc[0] - bottom['mean'].iloc[0]
                if len(top) and len(bottom) else np.nan
            ),
            'top_decile_hit_rate': top['hit_rate'].iloc[0] if len(top) else np.nan,
            'mean_spearman_ic': ic['mean_ic'].iloc[0] if len(ic) else np.nan,
            'ic_positive_rate': ic['ic_positive_rate'].iloc[0] if len(ic) else np.nan,
            'emerging_entry_alpha': emerging['mean'].iloc[0] if len(emerging) else np.nan,
            'emerging_entry_count': int(emerging['count'].iloc[0]) if len(emerging) else 0,
        })
    return pd.DataFrame(rows)



def entry_signal_study(
    prepared: pd.DataFrame,
    entries: pd.DataFrame,
    horizons: Iterable[int] = (5, 20, 60),
) -> pd.DataFrame:
    """Evaluate causal entry events against forward absolute/relative returns."""
    if entries is None or entries.empty:
        return pd.DataFrame(columns=['scope','horizon','metric','mean','median','hit_rate','count'])

    e = entries.copy()
    e['date'] = pd.to_datetime(e['entry_date']).dt.normalize()
    cols = ['date','ticker','entry_reason']
    e = e[cols].drop_duplicates(['date','ticker'])
    metrics = []
    for h in horizons:
        metrics.extend([
            (h,'absolute',f'fwd_{h}'),
            (h,'market_alpha',f'alpha_market_{h}'),
            (h,'sector_alpha',f'alpha_sector_{h}'),
        ])
    needed = ['date','ticker'] + sorted({m[2] for m in metrics})
    x = e.merge(prepared[[c for c in needed if c in prepared.columns]], on=['date','ticker'], how='left')

    rows=[]
    for scope,g0 in [('ALL',x)] + [(str(reason),g) for reason,g in x.groupby('entry_reason')]:
        for h,label,col in metrics:
            if col not in g0.columns:
                continue
            g = g0.dropna(subset=[col])
            rows.append({
                'scope': scope,
                'horizon': h,
                'metric': label,
                'mean': g[col].mean() if len(g) else np.nan,
                'median': g[col].median() if len(g) else np.nan,
                'hit_rate': (g[col] > 0).mean() if len(g) else np.nan,
                'count': len(g),
            })
    return pd.DataFrame(rows)


def score_comparison_study(
    prepared: pd.DataFrame,
    horizons: Iterable[int] = (5, 20, 60),
    scores: Iterable[str] = ('rs_score','real_strength_score','leadership_score'),
) -> pd.DataFrame:
    """Compare raw RS with V4 real strength on identical dates/universe.

    Spearman IC is calculated date-by-date against market- and sector-relative
    forward returns. This is a challenger diagnostic, not an alpha claim.
    """
    rows = []
    for score in scores:
        if score not in prepared.columns:
            continue
        for h in horizons:
            for metric, target in [
                ('market_alpha', f'alpha_market_{h}'),
                ('sector_alpha', f'alpha_sector_{h}'),
            ]:
                if target not in prepared.columns:
                    continue
                daily = []
                cols = ['date', score, target]
                for dt, g in prepared[cols].dropna().groupby('date'):
                    if len(g) < 20:
                        continue
                    ic = g[score].corr(g[target], method='spearman')
                    if pd.notna(ic):
                        daily.append(ic)
                s = pd.Series(daily, dtype=float)
                rows.append({
                    'score': score,
                    'horizon': h,
                    'metric': metric,
                    'mean_ic': s.mean() if len(s) else np.nan,
                    'median_ic': s.median() if len(s) else np.nan,
                    'ic_positive_rate': (s > 0).mean() if len(s) else np.nan,
                    'ic_std': s.std() if len(s) else np.nan,
                    'days': len(s),
                })
    return pd.DataFrame(rows)


def run_research_suite(panel: pd.DataFrame, horizons: Iterable[int] = (5, 20, 60)) -> dict[str, pd.DataFrame]:
    prepared = add_forward_returns(panel, horizons=horizons)
    deciles = decile_study(prepared, horizons=horizons)
    stages = stage_entry_study(prepared, horizons=horizons)
    ic_daily, ic_summary = information_coefficient(prepared, horizons=horizons)
    summary = research_summary(deciles, stages, ic_summary, horizons=horizons)
    score_comparison = score_comparison_study(prepared, horizons=horizons)
    return {
        'prepared': prepared,
        'deciles': deciles,
        'stage_entries': stages,
        'ic_daily': ic_daily,
        'ic_summary': ic_summary,
        'summary': summary,
        'score_comparison': score_comparison,
    }
