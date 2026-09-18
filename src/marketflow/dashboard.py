from __future__ import annotations

from pathlib import Path
import html
import math
import pandas as pd


DISPLAY = {
    'ticker': 'Mã',
    'sector': 'Ngành',
    'leadership_score': 'Leadership',
    'acceleration': 'Δ5 phiên',
    'rs_score': 'Sức mạnh tương đối',
    'flow_score': 'Dòng tiền',
    'trend_score': 'Chất lượng xu hướng',
    'sector_score': 'Sức mạnh ngành',
    'stage': 'Giai đoạn',
    'members': 'Số mã',
}


def _table(df: pd.DataFrame, columns: list[str], n=20) -> str:
    cols = [c for c in columns if c in df.columns]
    d = df[cols].head(n).copy()
    for c in d.select_dtypes(include='number').columns:
        d[c] = d[c].map(lambda v: '' if pd.isna(v) else f'{v:,.1f}')
    d = d.rename(columns={k: v for k, v in DISPLAY.items() if k in d.columns})
    return d.to_html(index=False, border=0, classes='data', escape=True)


def _rotation_svg(sectors: pd.DataFrame, width=980, height=460) -> str:
    if sectors.empty:
        return "<div class='muted'>Chưa đủ dữ liệu ngành.</div>"

    left, right, top, bottom = 62, 22, 34, 48
    pw, ph = width-left-right, height-top-bottom
    x0 = left + pw * 0.70
    y0 = top + ph * 0.50

    def sx(v):
        v = min(100.0, max(0.0, float(v)))
        return left + pw * v / 100.0

    def sy(v):
        v = min(30.0, max(-30.0, float(v)))
        return top + ph * (30.0-v) / 60.0

    chunks = [
        f"<svg viewBox='0 0 {width} {height}' class='rotation' role='img' aria-label='Bản đồ luân chuyển ngành'>",
        f"<line x1='{left}' y1='{y0:.1f}' x2='{width-right}' y2='{y0:.1f}' class='axis'/>",
        f"<line x1='{x0:.1f}' y1='{top}' x2='{x0:.1f}' y2='{height-bottom}' class='axis'/>",
        f"<text x='{width-right-8}' y='{height-12}' text-anchor='end' class='axis-label'>Sức mạnh ngành →</text>",
        f"<text x='{left}' y='18' class='axis-label'>Tăng tốc ↑</text>",
        f"<text x='{x0+10:.1f}' y='{top+18}' class='quad'>DẪN DẮT + TĂNG TỐC</text>",
        f"<text x='{left+10}' y='{top+18}' class='quad'>ĐANG HÌNH THÀNH</text>",
        f"<text x='{x0+10:.1f}' y='{height-bottom-10}' class='quad'>MẠNH NHƯNG HẠ NHIỆT</text>",
        f"<text x='{left+10}' y='{height-bottom-10}' class='quad'>YẾU / SUY GIẢM</text>",
    ]
    for _, row in sectors.head(24).iterrows():
        x, y = sx(row['sector_score']), sy(row.get('acceleration', 0) or 0)
        n = max(1.0, float(row.get('members', 1) or 1))
        radius = min(18, 6 + 2.2 * math.sqrt(n))
        name = html.escape(str(row['sector']))
        score = float(row['sector_score'])
        acc = float(row.get('acceleration', 0) or 0)
        cls = 'bubble hot' if score >= 70 and acc > 0 else ('bubble strong' if score >= 70 else ('bubble emerging' if acc > 5 else 'bubble'))
        chunks.append(
            f"<g><circle cx='{x:.1f}' cy='{y:.1f}' r='{radius:.1f}' class='{cls}'>"
            f"<title>{name}: strength {score:.1f}, Δ {acc:.1f}, {int(n)} mã</title></circle>"
            f"<text x='{x+radius+4:.1f}' y='{y+4:.1f}' class='bubble-label'>{name}</text></g>"
        )
    chunks.append("</svg>")
    return ''.join(chunks)


def render_dashboard(latest: pd.DataFrame, regime_latest: dict, out_path: str | Path):
    leaders = latest.sort_values('leadership_score', ascending=False)
    emerging = latest[latest['stage']=='EMERGING'].sort_values('acceleration', ascending=False)
    sectors = (latest.groupby('sector', as_index=False)
               .agg(sector_score=('sector_score','median'), acceleration=('acceleration','median'),
                    members=('ticker','count'))
               .sort_values(['sector_score','acceleration'], ascending=False))
    fading = latest[latest['stage']=='FADING'].sort_values('acceleration')
    dt = str(pd.to_datetime(latest['date'].max()).date())

    regime_map = {'RISK_ON':'RISK-ON','DEFENSIVE':'PHÒNG THỦ','NEUTRAL':'TRUNG TÍNH'}
    regime_text = regime_map.get(str(regime_latest.get('regime','NA')), str(regime_latest.get('regime','NA')))
    coverage = int(regime_latest.get('coverage_stocks', len(latest)) or 0)
    reference = int(regime_latest.get('coverage_reference', coverage) or 0)
    coverage_pct = 100*coverage/reference if reference else float('nan')

    css = """
    :root{--bg:#08101f;--panel:#111b31;--panel2:#0d1729;--line:#273653;--text:#edf2ff;--muted:#93a4c4;--accent:#67e8f9;--good:#4ade80;--warn:#fbbf24;--bad:#fb7185}
    *{box-sizing:border-box} body{font-family:Inter,ui-sans-serif,system-ui,-apple-system,Arial,sans-serif;background:var(--bg);color:var(--text);margin:0;padding:28px;max-width:1600px;margin-inline:auto}
    h1{font-size:30px;letter-spacing:-.03em;margin:0 0 6px} h2{font-size:17px;margin:0 0 12px}.muted{color:var(--muted);line-height:1.5}
    .topline{display:flex;justify-content:space-between;gap:16px;align-items:flex-end}.tag{font-size:12px;padding:5px 9px;border:1px solid var(--line);border-radius:999px;color:var(--muted)}
    .grid{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:20px 0}.card,.panel{background:linear-gradient(180deg,var(--panel),var(--panel2));border:1px solid var(--line);border-radius:14px;padding:18px}
    .big{font-size:32px;font-weight:750;letter-spacing:-.04em;margin-top:5px}.section{margin-top:22px}.two{display:grid;grid-template-columns:1fr 1fr;gap:18px}
    table.data{width:100%;border-collapse:collapse;background:transparent;font-size:13px}table.data th,table.data td{padding:9px 10px;border-bottom:1px solid rgba(39,54,83,.75);text-align:right;white-space:nowrap}
    table.data th:first-child,table.data td:first-child,table.data th:nth-child(2),table.data td:nth-child(2){text-align:left}table.data th{color:#aebcda;font-weight:600;background:rgba(255,255,255,.018)}
    .rotation{width:100%;height:auto;background:rgba(255,255,255,.015);border-radius:12px}.axis{stroke:#40506f;stroke-width:1;stroke-dasharray:5 5}.axis-label,.quad{fill:#8192b4;font-size:12px}.quad{font-size:10px;letter-spacing:.08em}
    .bubble{fill:#7586a8;fill-opacity:.65;stroke:#b8c5df;stroke-opacity:.5}.bubble.emerging{fill:var(--warn)}.bubble.strong{fill:var(--accent)}.bubble.hot{fill:var(--good)}
    .bubble-label{fill:#dce5f7;font-size:10px;paint-order:stroke;stroke:#08101f;stroke-width:3px;stroke-linejoin:round}.note{font-size:12px;color:var(--muted);margin-top:10px}
    @media(max-width:1000px){.grid,.two{grid-template-columns:1fr 1fr}}@media(max-width:650px){body{padding:16px}.grid,.two{grid-template-columns:1fr}.topline{align-items:flex-start;flex-direction:column}}
    """
    rotation = _rotation_svg(sectors)
    html_doc = f"""<!doctype html><html lang='vi'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width'><title>Vietnam Market Flow</title><style>{css}</style></head><body>
    <div class='topline'><div><h1>Vietnam Market Flow</h1><div class='muted'>Hệ thống độc lập: sức mạnh tương đối + dòng tiền + chất lượng xu hướng + xác nhận ngành</div></div><div class='tag'>Dữ liệu đến {html.escape(dt)}</div></div>
    <div class='grid'>
      <div class='card'><div class='muted'>Trạng thái thị trường</div><div class='big'>{html.escape(regime_text)}</div></div>
      <div class='card'><div class='muted'>Điểm thị trường</div><div class='big'>{regime_latest.get('market_score', float('nan')):.1f}</div></div>
      <div class='card'><div class='muted'>Độ rộng trên MA20</div><div class='big'>{100*regime_latest.get('breadth_ma20', float('nan')):.1f}%</div></div>
      <div class='card'><div class='muted'>Độ phủ dữ liệu</div><div class='big'>{coverage}/{reference}</div><div class='muted'>{coverage_pct:.0f}% universe tham chiếu</div></div>
    </div>

    <div class='panel section'><h2>Bản đồ luân chuyển ngành</h2>{rotation}<div class='note'>Trục ngang = sức mạnh ngành; trục dọc = thay đổi điểm trong 5 phiên. Kích thước bong bóng = số mã đủ điều kiện trong ngành.</div></div>

    <div class='two section'>
      <div class='panel'><h2>Cổ phiếu dẫn dắt</h2>{_table(leaders,['ticker','sector','leadership_score','acceleration','rs_score','flow_score','trend_score','stage'],20)}</div>
      <div class='panel'><h2>Đang tăng tốc</h2>{_table(emerging,['ticker','sector','leadership_score','acceleration','flow_score','sector_score','stage'],20)}</div>
    </div>
    <div class='panel section'><h2>Ngành theo sức mạnh</h2>{_table(sectors,['sector','sector_score','acceleration','members'],30)}</div>
    <div class='panel section'><h2>Suy yếu / mất động lượng</h2>{_table(fading,['ticker','sector','leadership_score','acceleration','rs_score','flow_score','stage'],20)}</div>
    <div class='note section'>“Dòng tiền” trong V1 là proxy từ giá, khối lượng và giá trị giao dịch; không phải số liệu mua ròng của tổ chức. Trọng số hiện là giả thuyết nghiên cứu và sẽ được hiệu chỉnh bằng backtest ngoài mẫu.</div>
    </body></html>"""
    Path(out_path).write_text(html_doc, encoding='utf-8')
