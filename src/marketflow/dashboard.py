from __future__ import annotations

from pathlib import Path
import html
import pandas as pd


def _table(df: pd.DataFrame, columns: list[str], n=20) -> str:
    d = df[columns].head(n).copy()
    for c in d.select_dtypes(include='number').columns:
        d[c] = d[c].map(lambda v: '' if pd.isna(v) else f'{v:,.1f}')
    return d.to_html(index=False, border=0, classes='data')


def render_dashboard(latest: pd.DataFrame, regime_latest: dict, out_path: str | Path):
    leaders = latest.sort_values('leadership_score', ascending=False)
    emerging = latest[latest['stage']=='EMERGING'].sort_values('acceleration', ascending=False)
    sectors = (latest.groupby('sector', as_index=False)
               .agg(sector_score=('sector_score','median'), acceleration=('acceleration','median'),
                    members=('ticker','count'))
               .sort_values('sector_score', ascending=False))
    fading = latest[latest['stage']=='FADING'].sort_values('acceleration')
    dt = str(pd.to_datetime(latest['date'].max()).date())
    css = """
    body{font-family:Inter,Arial,sans-serif;background:#0b1020;color:#e8ecf3;margin:0;padding:28px}
    h1,h2{margin:0 0 12px}.muted{color:#8d99ae}.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:18px 0}
    .card{background:#121a2f;border:1px solid #26314d;border-radius:14px;padding:18px}.big{font-size:34px;font-weight:700}
    .pill{display:inline-block;padding:6px 10px;border-radius:999px;background:#1c2948}.section{margin-top:24px}
    table.data{width:100%;border-collapse:collapse;background:#121a2f;border-radius:12px;overflow:hidden}
    table.data th,table.data td{padding:10px 12px;border-bottom:1px solid #25314c;text-align:right}table.data th:first-child,table.data td:first-child{text-align:left}
    table.data th{color:#aab6d0;background:#151f38}.two{display:grid;grid-template-columns:1fr 1fr;gap:18px}
    @media(max-width:900px){.grid,.two{grid-template-columns:1fr}}
    """
    html_doc = f"""<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width'><title>Vietnam Market Flow</title><style>{css}</style></head><body>
    <h1>Vietnam Market Flow Engine</h1><div class='muted'>Independent market-flow & leadership model • {html.escape(dt)}</div>
    <div class='grid'>
      <div class='card'><div class='muted'>Market regime</div><div class='big'>{html.escape(str(regime_latest.get('regime','NA')))}</div></div>
      <div class='card'><div class='muted'>Market score</div><div class='big'>{regime_latest.get('market_score', float('nan')):.1f}</div></div>
      <div class='card'><div class='muted'>Breadth &gt; MA20</div><div class='big'>{100*regime_latest.get('breadth_ma20', float('nan')):.1f}%</div></div>
      <div class='card'><div class='muted'>Coverage</div><div class='big'>{int(regime_latest.get('coverage_stocks', 0))}/{int(regime_latest.get('coverage_reference', 0))}</div></div>
    </div>
    <div class='section'><h2>Sector rotation</h2>{_table(sectors,['sector','sector_score','acceleration','members'],20)}</div>
    <div class='two section'><div><h2>Top leaders</h2>{_table(leaders,['ticker','sector','leadership_score','acceleration','rs_score','flow_score','trend_score','stage'],20)}</div>
    <div><h2>Emerging</h2>{_table(emerging,['ticker','sector','leadership_score','acceleration','flow_score','stage'],20)}</div></div>
    <div class='section'><h2>Fading / weak</h2>{_table(fading,['ticker','sector','leadership_score','acceleration','rs_score','flow_score','stage'],20)}</div>
    </body></html>"""
    Path(out_path).write_text(html_doc, encoding='utf-8')
