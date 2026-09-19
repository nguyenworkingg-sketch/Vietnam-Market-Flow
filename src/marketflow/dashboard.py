from __future__ import annotations

from pathlib import Path
import html
import math
import numpy as np
import pandas as pd


DISPLAY = {
    'ticker': 'Mã',
    'sector': 'Ngành',
    'leadership_score': 'Điểm dẫn dắt',
    'acceleration': 'Tăng tốc 5 phiên',
    'rs_score': 'Sức mạnh tương đối',
    'flow_score': 'Dòng tiền',
    'trend_score': 'Chất lượng xu hướng',
    'sector_score': 'Sức mạnh ngành',
    'stage': 'Giai đoạn',
    'members': 'Số mã',
}

STAGE_VI = {
    'LEADER': 'DẪN DẮT',
    'MATURE': 'TRƯỞNG THÀNH',
    'EMERGING': 'TĂNG TỐC',
    'FADING': 'SUY YẾU',
    'NEUTRAL': 'TRUNG TÍNH',
}


def _num(v, digits=1, suffix=''):
    try:
        if pd.isna(v):
            return '—'
        return f'{float(v):,.{digits}f}{suffix}'
    except Exception:
        return '—'


def _table(df: pd.DataFrame, columns: list[str], n=20) -> str:
    cols = [c for c in columns if c in df.columns]
    d = df[cols].head(n).copy()
    if d.empty:
        return "<div class='empty'>Chưa có dữ liệu phù hợp.</div>"
    for c in d.select_dtypes(include='number').columns:
        d[c] = d[c].map(lambda v: '' if pd.isna(v) else f'{v:,.1f}')
    if 'stage' in d.columns:
        d['stage'] = d['stage'].map(lambda x: STAGE_VI.get(str(x), str(x)))
    d = d.rename(columns={k: v for k, v in DISPLAY.items() if k in d.columns})
    return d.to_html(index=False, border=0, classes='data', escape=True)


def _rotation_svg(sectors: pd.DataFrame, width=980, height=440) -> str:
    if sectors.empty:
        return "<div class='empty'>Chưa đủ dữ liệu ngành.</div>"

    left, right, top, bottom = 58, 22, 34, 42
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
        f"<svg viewBox='0 0 {width} {height}' class='chart-svg rotation' role='img' aria-label='Bản đồ luân chuyển ngành'>",
        f"<rect x='{x0:.1f}' y='{top}' width='{width-right-x0:.1f}' height='{y0-top:.1f}' class='quad-fill q-good'/>",
        f"<rect x='{left}' y='{top}' width='{x0-left:.1f}' height='{y0-top:.1f}' class='quad-fill q-watch'/>",
        f"<rect x='{x0:.1f}' y='{y0:.1f}' width='{width-right-x0:.1f}' height='{height-bottom-y0:.1f}' class='quad-fill q-cool'/>",
        f"<rect x='{left}' y='{y0:.1f}' width='{x0-left:.1f}' height='{height-bottom-y0:.1f}' class='quad-fill q-weak'/>",
        f"<line x1='{left}' y1='{y0:.1f}' x2='{width-right}' y2='{y0:.1f}' class='axis'/>",
        f"<line x1='{x0:.1f}' y1='{top}' x2='{x0:.1f}' y2='{height-bottom}' class='axis'/>",
        f"<text x='{width-right-8}' y='{height-10}' text-anchor='end' class='axis-label'>Sức mạnh ngành →</text>",
        f"<text x='{left}' y='18' class='axis-label'>Tăng tốc ↑</text>",
        f"<text x='{x0+10:.1f}' y='{top+18}' class='quad'>DẪN DẮT + TĂNG TỐC</text>",
        f"<text x='{left+10}' y='{top+18}' class='quad'>ĐANG HÌNH THÀNH</text>",
        f"<text x='{x0+10:.1f}' y='{height-bottom-10}' class='quad'>MẠNH NHƯNG HẠ NHIỆT</text>",
        f"<text x='{left+10}' y='{height-bottom-10}' class='quad'>YẾU / SUY GIẢM</text>",
    ]
    for _, row in sectors.head(24).iterrows():
        x, y = sx(row['sector_score']), sy(row.get('acceleration', 0) or 0)
        n = max(1.0, float(row.get('members', 1) or 1))
        radius = min(17, 5.5 + 2.0 * math.sqrt(n))
        name = html.escape(str(row['sector']))
        score = float(row['sector_score'])
        acc = float(row.get('acceleration', 0) or 0)
        cls = 'bubble hot' if score >= 70 and acc > 0 else ('bubble strong' if score >= 70 else ('bubble emerging' if acc > 5 else 'bubble'))
        chunks.append(
            f"<g><circle cx='{x:.1f}' cy='{y:.1f}' r='{radius:.1f}' class='{cls}'>"
            f"<title>{name}: sức mạnh {score:.1f}, tăng tốc {acc:.1f}, {int(n)} mã</title></circle>"
            f"<text x='{x+radius+4:.1f}' y='{y+4:.1f}' class='bubble-label'>{name}</text></g>"
        )
    chunks.append("</svg>")
    return ''.join(chunks)


def _line_svg(df: pd.DataFrame, series: list[tuple[str, str, str]], width=920, height=270, days=60) -> str:
    if df is None or df.empty or 'date' not in df.columns:
        return "<div class='empty'>Chưa đủ dữ liệu lịch sử.</div>"
    x = df.copy()
    x['date'] = pd.to_datetime(x['date'])
    x = x.sort_values('date').tail(days)
    if x.empty:
        return "<div class='empty'>Chưa đủ dữ liệu lịch sử.</div>"

    left, right, top, bottom = 46, 20, 20, 34
    pw, ph = width-left-right, height-top-bottom
    dates = x['date'].drop_duplicates().sort_values().tolist()
    if len(dates) < 2:
        return "<div class='empty'>Cần thêm lịch sử để vẽ xu hướng.</div>"
    dmap = {d: i for i, d in enumerate(dates)}

    def sx(d):
        return left + pw * dmap[d] / max(1, len(dates)-1)

    def sy(v):
        v = min(100.0, max(0.0, float(v)))
        return top + ph * (100-v) / 100.0

    chunks = [f"<svg viewBox='0 0 {width} {height}' class='chart-svg' role='img'>"]
    for val in [0, 25, 50, 75, 100]:
        y = sy(val)
        chunks.append(f"<line x1='{left}' y1='{y:.1f}' x2='{width-right}' y2='{y:.1f}' class='gridline'/>")
        chunks.append(f"<text x='{left-8}' y='{y+4:.1f}' text-anchor='end' class='tick'>{val}</text>")

    for col, label, cls in series:
        if col not in x.columns:
            continue
        pts = []
        for _, row in x[['date', col]].dropna().iterrows():
            pts.append(f"{sx(row['date']):.1f},{sy(row[col]):.1f}")
        if len(pts) >= 2:
            chunks.append(f"<polyline points='{' '.join(pts)}' class='series {cls}'/>")

    tick_idx = sorted(set(np.linspace(0, len(dates)-1, min(6, len(dates))).astype(int).tolist()))
    for i in tick_idx:
        d = dates[i]
        chunks.append(f"<text x='{sx(d):.1f}' y='{height-10}' text-anchor='middle' class='tick'>{pd.Timestamp(d).strftime('%d/%m')}</text>")

    lx = left
    for _, label, cls in series:
        chunks.append(f"<line x1='{lx}' y1='10' x2='{lx+18}' y2='10' class='series {cls}'/>")
        chunks.append(f"<text x='{lx+23}' y='14' class='legend'>{html.escape(label)}</text>")
        lx += 145
    chunks.append("</svg>")
    return ''.join(chunks)


def _histogram_svg(values: pd.Series, width=520, height=250, bins=10) -> str:
    vals = pd.to_numeric(values, errors='coerce').dropna().clip(0, 100)
    if vals.empty:
        return "<div class='empty'>Chưa có dữ liệu.</div>"
    counts, edges = np.histogram(vals, bins=bins, range=(0, 100))
    left, right, top, bottom = 42, 18, 18, 34
    pw, ph = width-left-right, height-top-bottom
    mx = max(1, int(counts.max()))
    bw = pw / bins
    chunks = [f"<svg viewBox='0 0 {width} {height}' class='chart-svg' role='img' aria-label='Phân phối điểm dẫn dắt'>"]
    for i, count in enumerate(counts):
        h = ph * count / mx
        x = left + i*bw + 3
        y = top + ph - h
        chunks.append(f"<rect x='{x:.1f}' y='{y:.1f}' width='{max(2,bw-6):.1f}' height='{h:.1f}' class='histbar'><title>{edges[i]:.0f}–{edges[i+1]:.0f}: {int(count)} mã</title></rect>")
    chunks.append(f"<line x1='{left}' y1='{top+ph}' x2='{width-right}' y2='{top+ph}' class='axis-solid'/>")
    for v in [0, 20, 40, 60, 80, 100]:
        x = left + pw*v/100
        chunks.append(f"<text x='{x:.1f}' y='{height-10}' text-anchor='middle' class='tick'>{v}</text>")
    chunks.append("</svg>")
    return ''.join(chunks)


def _scatter_svg(latest: pd.DataFrame, width=760, height=330) -> str:
    if latest.empty:
        return "<div class='empty'>Chưa có dữ liệu.</div>"
    x = latest.dropna(subset=['leadership_score', 'acceleration']).copy()
    if x.empty:
        return "<div class='empty'>Chưa có dữ liệu.</div>"
    left, right, top, bottom = 52, 24, 24, 42
    pw, ph = width-left-right, height-top-bottom

    def sx(v):
        return left + pw * min(100, max(0, float(v))) / 100

    def sy(v):
        v = min(30, max(-30, float(v)))
        return top + ph * (30-v)/60

    chunks = [f"<svg viewBox='0 0 {width} {height}' class='chart-svg' role='img' aria-label='Điểm dẫn dắt và tăng tốc'>"]
    for v in [0,25,50,75,100]:
        xx=sx(v)
        chunks.append(f"<line x1='{xx:.1f}' y1='{top}' x2='{xx:.1f}' y2='{height-bottom}' class='gridline'/>")
        chunks.append(f"<text x='{xx:.1f}' y='{height-12}' text-anchor='middle' class='tick'>{v}</text>")
    for v in [-30,-15,0,15,30]:
        yy=sy(v)
        chunks.append(f"<line x1='{left}' y1='{yy:.1f}' x2='{width-right}' y2='{yy:.1f}' class='gridline'/>")
        chunks.append(f"<text x='{left-8}' y='{yy+4:.1f}' text-anchor='end' class='tick'>{v}</text>")
    chunks.append(f"<text x='{width-right}' y='{height-2}' text-anchor='end' class='axis-label'>Điểm dẫn dắt →</text>")
    chunks.append(f"<text x='{left}' y='14' class='axis-label'>Tăng tốc ↑</text>")

    label_idx = set(x.sort_values(['leadership_score','acceleration'], ascending=False).head(16).index)
    for idx,row in x.iterrows():
        score=float(row['leadership_score']); acc=float(row['acceleration'])
        flow=float(row.get('flow_score',50) if pd.notna(row.get('flow_score',np.nan)) else 50)
        r=3.5+max(0,min(100,flow))/100*5.5
        stage=str(row.get('stage','NEUTRAL'))
        cls='dot leader' if stage in ('LEADER','MATURE') else ('dot emerging' if stage=='EMERGING' else ('dot fading' if stage=='FADING' else 'dot'))
        name=html.escape(str(row['ticker']))
        sector=html.escape(str(row.get('sector','')))
        chunks.append(f"<circle cx='{sx(score):.1f}' cy='{sy(acc):.1f}' r='{r:.1f}' class='{cls}'><title>{name} · {sector} · điểm {score:.1f} · tăng tốc {acc:.1f} · dòng tiền {flow:.1f}</title></circle>")
        if idx in label_idx:
            chunks.append(f"<text x='{sx(score)+r+3:.1f}' y='{sy(acc)+3:.1f}' class='point-label'>{name}</text>")
    chunks.append("</svg>")
    return ''.join(chunks)


def _sector_bars(sectors: pd.DataFrame, n=10) -> str:
    if sectors.empty:
        return "<div class='empty'>Chưa có dữ liệu ngành.</div>"
    rows=[]
    for _,r in sectors.head(n).iterrows():
        score=float(r['sector_score'])
        acc=float(r.get('acceleration',0) or 0)
        acc_cls='pos' if acc>0 else ('neg' if acc<0 else '')
        rows.append(
            "<div class='bar-row'>"
            f"<div class='bar-name'>{html.escape(str(r['sector']))}</div>"
            f"<div class='bar-track'><div class='bar-fill' style='width:{max(0,min(100,score)):.1f}%'></div></div>"
            f"<div class='bar-score'>{score:.1f}</div>"
            f"<div class='bar-delta {acc_cls}'>{acc:+.1f}</div>"
            "</div>"
        )
    return ''.join(rows)


def _sector_rank_leaders(latest: pd.DataFrame, sectors: pd.DataFrame, stocks_per_sector: int = 5) -> str:
    if latest.empty or sectors.empty:
        return "<div class='empty'>Chưa có dữ liệu xếp hạng ngành.</div>"

    ranked = sectors.sort_values(['sector_score','acceleration'], ascending=False).reset_index(drop=True)
    ranked['sector_rank'] = ranked.index + 1
    chunks = ["<div class='sector-rank-list'>"]

    for _, sec in ranked.iterrows():
        sector = str(sec['sector'])
        stocks = (latest[latest['sector'].astype(str).eq(sector)]
                  .sort_values(['leadership_score','acceleration'], ascending=False)
                  .head(stocks_per_sector)
                  .copy())
        if stocks.empty:
            continue

        leader = stocks.iloc[0]
        score = float(sec['sector_score'])
        acc = float(sec.get('acceleration', 0) or 0)
        members = int(sec.get('members', len(stocks)) or 0)
        rank = int(sec['sector_rank'])
        acc_cls = 'pos' if acc > 0 else ('neg' if acc < 0 else '')
        rank_cls = 'rank-top' if rank <= 3 else ('rank-mid' if rank <= 8 else 'rank-low')
        leader_stage = STAGE_VI.get(str(leader.get('stage','NEUTRAL')), str(leader.get('stage','NEUTRAL')))

        stock_chips = []
        for i, (_, row) in enumerate(stocks.iterrows(), start=1):
            ticker = html.escape(str(row['ticker']))
            lscore = float(row.get('leadership_score', np.nan))
            delta = float(row.get('acceleration', 0) or 0)
            stage = STAGE_VI.get(str(row.get('stage','NEUTRAL')), str(row.get('stage','NEUTRAL')))
            chip_cls = 'stock-chip best' if i == 1 else 'stock-chip'
            stock_chips.append(
                f"<div class='{chip_cls}' title='{stage} · tăng tốc {delta:+.1f}'>"
                f"<span class='chip-rank'>#{i}</span><b>{ticker}</b>"
                f"<span class='chip-score'>{lscore:.1f}</span>"
                "</div>"
            )

        chunks.append(
            "<details class='sector-rank-row'>"
            "<summary>"
            f"<span class='sector-rank {rank_cls}'>#{rank}</span>"
            f"<span class='sector-title'>{html.escape(sector)}<small>{members} mã đủ điều kiện</small></span>"
            f"<span class='sector-meter'><span class='sector-meter-fill' style='width:{max(0,min(100,score)):.1f}%'></span></span>"
            f"<span class='sector-strength'>{score:.1f}</span>"
            f"<span class='sector-acc {acc_cls}'>{acc:+.1f}</span>"
            f"<span class='sector-leader'><b>{html.escape(str(leader['ticker']))}</b><small>{float(leader['leadership_score']):.1f} · {leader_stage}</small></span>"
            "</summary>"
            "<div class='sector-stocks'>"
            "<div class='sector-stocks-label'>Xếp hạng cổ phiếu trong ngành</div>"
            f"<div class='stock-chips'>{''.join(stock_chips)}</div>"
            "</div>"
            "</details>"
        )

    chunks.append("</div>")
    return ''.join(chunks)


def _sector_history_svg(sector_history: pd.DataFrame | None, top_sectors: list[str], width=920, height=290, days=40) -> str:
    if sector_history is None or sector_history.empty or not top_sectors:
        return "<div class='empty'>Chưa đủ lịch sử ngành.</div>"
    x=sector_history.copy()
    x['date']=pd.to_datetime(x['date'])
    x=x[x['sector'].isin(top_sectors)].sort_values('date')
    if x.empty:
        return "<div class='empty'>Chưa đủ lịch sử ngành.</div>"
    dates=sorted(x['date'].unique())[-days:]
    x=x[x['date'].isin(dates)]
    left,right,top,bottom=46,20,24,38
    pw,ph=width-left-right,height-top-bottom
    date_list=sorted(pd.to_datetime(x['date'].unique()).tolist())
    if len(date_list)<2:
        return "<div class='empty'>Cần thêm lịch sử ngành.</div>"
    dmap={d:i for i,d in enumerate(date_list)}
    def sx(d): return left+pw*dmap[pd.Timestamp(d)]/max(1,len(date_list)-1)
    def sy(v): return top+ph*(100-min(100,max(0,float(v))))/100
    classes=['s1','s2','s3','s4','s5']
    chunks=[f"<svg viewBox='0 0 {width} {height}' class='chart-svg' role='img'>"]
    for v in [0,25,50,75,100]:
        yy=sy(v); chunks.append(f"<line x1='{left}' y1='{yy:.1f}' x2='{width-right}' y2='{yy:.1f}' class='gridline'/><text x='{left-8}' y='{yy+4:.1f}' text-anchor='end' class='tick'>{v}</text>")
    for k,sector in enumerate(top_sectors[:5]):
        g=x[x['sector']==sector].dropna(subset=['sector_score'])
        pts=' '.join(f"{sx(r['date']):.1f},{sy(r['sector_score']):.1f}" for _,r in g.iterrows())
        if pts:
            chunks.append(f"<polyline points='{pts}' class='series {classes[k]}'/>")
    tick_idx=sorted(set(np.linspace(0,len(date_list)-1,min(5,len(date_list))).astype(int).tolist()))
    for i in tick_idx:
        d=date_list[i]; chunks.append(f"<text x='{sx(d):.1f}' y='{height-10}' text-anchor='middle' class='tick'>{pd.Timestamp(d).strftime('%d/%m')}</text>")
    lx=left
    for k,sector in enumerate(top_sectors[:5]):
        chunks.append(f"<line x1='{lx}' y1='11' x2='{lx+18}' y2='11' class='series {classes[k]}'/><text x='{lx+23}' y='15' class='legend'>{html.escape(str(sector))}</text>")
        lx+=max(130,20+len(str(sector))*7)
    chunks.append("</svg>")
    return ''.join(chunks)


def _stage_distribution(latest: pd.DataFrame) -> str:
    counts=latest['stage'].fillna('NEUTRAL').astype(str).value_counts()
    total=max(1,int(counts.sum()))
    order=['LEADER','MATURE','EMERGING','NEUTRAL','FADING']
    parts=[]
    legend=[]
    for stage in order:
        n=int(counts.get(stage,0))
        if n<=0:
            continue
        pct=100*n/total
        parts.append(f"<div class='stage-seg st-{stage.lower()}' style='width:{pct:.2f}%' title='{STAGE_VI.get(stage,stage)}: {n} mã ({pct:.1f}%)'></div>")
        legend.append(f"<div class='stage-item'><span class='stage-dot st-{stage.lower()}'></span><span>{STAGE_VI.get(stage,stage)}</span><b>{n}</b><small>{pct:.0f}%</small></div>")
    return "<div class='stage-stack'>"+''.join(parts)+"</div><div class='stage-legend'>"+''.join(legend)+"</div>"


def _backtest_decile_svg(deciles: pd.DataFrame | None, horizon=20, width=620, height=270) -> str:
    if deciles is None or deciles.empty:
        return "<div class='empty'>Backtest chưa đủ dữ liệu.</div>"
    d=deciles[(deciles['horizon']==horizon)&(deciles['metric']=='market_alpha')].sort_values('decile').copy()
    if d.empty:
        return "<div class='empty'>Backtest chưa đủ dữ liệu.</div>"
    vals=pd.to_numeric(d['mean'],errors='coerce').fillna(0)*100
    lim=max(1.0,float(np.nanmax(np.abs(vals)))*1.15)
    left,right,top,bottom=42,18,18,38
    pw,ph=width-left-right,height-top-bottom
    zero=top+ph/2
    bw=pw/max(1,len(d))
    chunks=[f"<svg viewBox='0 0 {width} {height}' class='chart-svg' role='img' aria-label='Alpha theo decile'>",
            f"<line x1='{left}' y1='{zero:.1f}' x2='{width-right}' y2='{zero:.1f}' class='axis-solid'/>"]
    for i,(_,r) in enumerate(d.iterrows()):
        v=float(r['mean'])*100
        h=abs(v)/lim*(ph/2)
        yy=zero-h if v>=0 else zero
        cls='alpha-pos' if v>=0 else 'alpha-neg'
        xx=left+i*bw+4
        chunks.append(f"<rect x='{xx:.1f}' y='{yy:.1f}' width='{max(4,bw-8):.1f}' height='{h:.1f}' class='{cls}'><title>Decile {int(r['decile'])}: {v:+.2f}%</title></rect>")
        chunks.append(f"<text x='{xx+(bw-8)/2:.1f}' y='{height-12}' text-anchor='middle' class='tick'>D{int(r['decile'])}</text>")
    chunks.append("</svg>")
    return ''.join(chunks)


def _ic_svg(ic_daily: pd.DataFrame | None, horizon=20, width=620, height=270, days=80) -> str:
    if ic_daily is None or ic_daily.empty:
        return "<div class='empty'>IC chưa đủ dữ liệu.</div>"
    d=ic_daily[(ic_daily['horizon']==horizon)&(ic_daily['metric']=='market_alpha')].copy()
    d['date']=pd.to_datetime(d['date'])
    d=d.sort_values('date').tail(days)
    if len(d)<2:
        return "<div class='empty'>IC chưa đủ dữ liệu.</div>"
    left,right,top,bottom=46,18,18,38
    pw,ph=width-left-right,height-top-bottom
    lim=max(.15,min(.8,float(np.nanmax(np.abs(pd.to_numeric(d['ic'],errors='coerce'))))*1.15))
    dates=d['date'].tolist()
    def sx(i): return left+pw*i/max(1,len(dates)-1)
    def sy(v): return top+ph*(lim-float(v))/(2*lim)
    zero=sy(0)
    chunks=[f"<svg viewBox='0 0 {width} {height}' class='chart-svg' role='img'>",
            f"<line x1='{left}' y1='{zero:.1f}' x2='{width-right}' y2='{zero:.1f}' class='axis-solid'/>"]
    pts=' '.join(f"{sx(i):.1f},{sy(r['ic']):.1f}" for i,(_,r) in enumerate(d.iterrows()) if pd.notna(r['ic']))
    chunks.append(f"<polyline points='{pts}' class='series cyan'/>")
    tick_idx=sorted(set(np.linspace(0,len(dates)-1,min(5,len(dates))).astype(int).tolist()))
    for i in tick_idx:
        chunks.append(f"<text x='{sx(i):.1f}' y='{height-10}' text-anchor='middle' class='tick'>{pd.Timestamp(dates[i]).strftime('%d/%m')}</text>")
    chunks.append(f"<text x='{left-8}' y='{sy(lim)+4:.1f}' text-anchor='end' class='tick'>{lim:.2f}</text>")
    chunks.append(f"<text x='{left-8}' y='{sy(-lim)+4:.1f}' text-anchor='end' class='tick'>-{lim:.2f}</text>")
    chunks.append("</svg>")
    return ''.join(chunks)


def _backtest_cards(summary: pd.DataFrame | None) -> str:
    if summary is None or summary.empty:
        return "<div class='empty'>Backtest đang tích lũy thêm forward observations.</div>"
    cards=[]
    for _,r in summary.sort_values('horizon').iterrows():
        h=int(r['horizon'])
        spread=r.get('top_bottom_spread',np.nan)
        hit=r.get('top_decile_hit_rate',np.nan)
        ic=r.get('mean_spearman_ic',np.nan)
        cards.append(
            "<div class='mini-card'>"
            f"<div class='mini-title'>{h} phiên</div>"
            f"<div class='mini-grid'><span>Spread D10–D1</span><b>{_num(100*spread,2,'%')}</b>"
            f"<span>Hit-rate D10</span><b>{_num(100*hit,1,'%')}</b>"
            f"<span>Spearman IC</span><b>{_num(ic,3)}</b></div>"
            "</div>"
        )
    return ''.join(cards)


def _signal_readout(latest: pd.DataFrame, regime_history: pd.DataFrame | None, sectors: pd.DataFrame) -> str:
    notes=[]
    leaders=int(latest['stage'].isin(['LEADER','MATURE']).sum())
    emerging=int((latest['stage']=='EMERGING').sum())
    fading=int((latest['stage']=='FADING').sum())
    notes.append(f"<li><b>{leaders}</b> mã đang ở nhóm dẫn dắt/trưởng thành; <b>{emerging}</b> mã đang tăng tốc và <b>{fading}</b> mã suy yếu.</li>")
    if not sectors.empty:
        r=sectors.iloc[0]
        notes.append(f"<li>Ngành có điểm sức mạnh cao nhất hiện tại là <b>{html.escape(str(r['sector']))}</b> ({float(r['sector_score']):.1f}), tăng tốc {float(r.get('acceleration',0) or 0):+.1f} điểm.</li>")
    if regime_history is not None and not regime_history.empty and 'market_score' in regime_history.columns:
        rh=regime_history.sort_values('date').dropna(subset=['market_score'])
        if len(rh)>=6:
            delta=float(rh.iloc[-1]['market_score'])-float(rh.iloc[-6]['market_score'])
            direction='tăng' if delta>0 else 'giảm'
            notes.append(f"<li>Điểm thị trường {direction} <b>{abs(delta):.1f}</b> điểm so với 5 phiên trước.</li>")
    return "<ul class='readout'>"+''.join(notes)+"</ul>"


def render_dashboard(
    latest: pd.DataFrame,
    regime_latest: dict,
    out_path: str | Path,
    scored_history: pd.DataFrame | None = None,
    regime_history: pd.DataFrame | None = None,
    sector_history: pd.DataFrame | None = None,
    backtest: dict[str, pd.DataFrame] | None = None,
):
    latest=latest.copy()
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
    breadth20=100*float(regime_latest.get('breadth_ma20',np.nan))
    breadth50=100*float(regime_latest.get('breadth_ma50',np.nan))
    leader_count=int(latest['stage'].isin(['LEADER','MATURE']).sum())

    rh=None
    if regime_history is not None and not regime_history.empty:
        rh=regime_history.copy()
        rh['breadth_ma20_pct']=pd.to_numeric(rh.get('breadth_ma20'),errors='coerce')*100
        rh['breadth_ma50_pct']=pd.to_numeric(rh.get('breadth_ma50'),errors='coerce')*100

    top_sectors=sectors.head(5)['sector'].astype(str).tolist()
    bt=backtest or {}

    css = """
    :root{
      --bg:#07101d;--panel:#0d1828;--panel2:#101d31;--line:#20314c;--text:#edf4ff;--muted:#8fa2bd;
      --cyan:#54d7ef;--green:#45d483;--amber:#f2bf55;--rose:#f0748b;--blue:#6ea8fe;--violet:#a78bfa
    }
    *{box-sizing:border-box}html{scroll-behavior:smooth}body{font-family:Inter,ui-sans-serif,system-ui,-apple-system,Segoe UI,Arial,sans-serif;background:var(--bg);color:var(--text);margin:0}
    .shell{max-width:1560px;margin:auto;padding:24px 28px 48px}.topline{display:flex;justify-content:space-between;gap:18px;align-items:flex-end}.eyebrow{font-size:11px;letter-spacing:.16em;text-transform:uppercase;color:var(--cyan);font-weight:750}
    h1{font-size:31px;letter-spacing:-.035em;margin:5px 0 5px}h2{font-size:17px;margin:0 0 13px;letter-spacing:-.01em}h3{font-size:13px;color:#c8d5e9;margin:0 0 10px}.muted,.note{color:var(--muted);line-height:1.5}.note{font-size:12px}.tag{font-size:12px;padding:6px 10px;border:1px solid var(--line);border-radius:999px;color:#b9c8dd;background:#0a1525}
    .nav{position:sticky;top:0;z-index:9;display:flex;gap:7px;flex-wrap:wrap;padding:10px 0;margin-top:15px;background:linear-gradient(180deg,var(--bg) 78%,transparent)}.nav a{text-decoration:none;color:#9fb1cb;border:1px solid var(--line);background:#0a1525;padding:7px 10px;border-radius:8px;font-size:12px}.nav a:hover{color:white;border-color:#3b577d}
    .kpis{display:grid;grid-template-columns:repeat(6,1fr);gap:12px;margin:10px 0 18px}.card,.panel,.mini-card{background:linear-gradient(180deg,var(--panel2),var(--panel));border:1px solid var(--line);border-radius:13px;box-shadow:0 8px 28px rgba(0,0,0,.10)}
    .card{padding:15px 16px}.card-label{color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.08em}.big{font-size:27px;font-weight:780;letter-spacing:-.04em;margin-top:5px}.sub{font-size:11px;color:var(--muted);margin-top:4px}
    .section{margin-top:18px}.panel{padding:17px}.grid-2{display:grid;grid-template-columns:1.15fr .85fr;gap:16px}.grid-even{display:grid;grid-template-columns:1fr 1fr;gap:16px}.grid-3{display:grid;grid-template-columns:repeat(3,1fr);gap:13px}
    .section-head{display:flex;justify-content:space-between;align-items:center;gap:10px;margin-bottom:10px}.section-kicker{font-size:11px;color:var(--cyan);letter-spacing:.13em;text-transform:uppercase;font-weight:700}.empty{color:var(--muted);font-size:13px;padding:24px 8px;text-align:center}
    .chart-svg{width:100%;height:auto;overflow:visible}.gridline{stroke:#20314c;stroke-width:1;stroke-dasharray:3 5}.axis,.axis-solid{stroke:#40516d;stroke-width:1}.axis{stroke-dasharray:5 5}.axis-label,.tick,.legend,.quad{fill:#8295b2;font-size:10px}.legend{fill:#adbad0}.quad{font-size:9px;letter-spacing:.08em}.quad-fill{opacity:.05}.q-good{fill:var(--green)}.q-watch{fill:var(--amber)}.q-cool{fill:var(--cyan)}.q-weak{fill:var(--rose)}
    .series{fill:none;stroke-width:2.2;vector-effect:non-scaling-stroke}.series.cyan{stroke:var(--cyan)}.series.green{stroke:var(--green)}.series.blue{stroke:var(--blue)}.series.amber{stroke:var(--amber)}.series.s1{stroke:var(--cyan)}.series.s2{stroke:var(--green)}.series.s3{stroke:var(--amber)}.series.s4{stroke:var(--violet)}.series.s5{stroke:var(--rose)}
    .bubble{fill:#7186a5;fill-opacity:.66;stroke:#c2cce0;stroke-opacity:.42}.bubble.emerging{fill:var(--amber)}.bubble.strong{fill:var(--cyan)}.bubble.hot{fill:var(--green)}.bubble-label,.point-label{fill:#dfe9f8;font-size:9px;paint-order:stroke;stroke:#07101d;stroke-width:3px;stroke-linejoin:round}
    .dot{fill:#7287a7;fill-opacity:.55}.dot.leader{fill:var(--green);fill-opacity:.76}.dot.emerging{fill:var(--amber);fill-opacity:.78}.dot.fading{fill:var(--rose);fill-opacity:.68}
    .histbar{fill:#4c8fcf;opacity:.8}.alpha-pos{fill:var(--green);opacity:.83}.alpha-neg{fill:var(--rose);opacity:.8}
    .bar-row{display:grid;grid-template-columns:minmax(120px,1fr) 2.2fr 46px 48px;align-items:center;gap:9px;padding:7px 0;border-bottom:1px solid rgba(32,49,76,.55)}.bar-name{font-size:12px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.bar-track{height:7px;border-radius:999px;background:#15243a;overflow:hidden}.bar-fill{height:100%;background:linear-gradient(90deg,#387fae,var(--cyan));border-radius:999px}.bar-score,.bar-delta{text-align:right;font-variant-numeric:tabular-nums;font-size:12px}.bar-delta.pos{color:var(--green)}.bar-delta.neg{color:var(--rose)}
    .sector-rank-list{display:flex;flex-direction:column;gap:7px}.sector-rank-row{margin:0;border:1px solid rgba(32,49,76,.72);border-radius:10px;background:rgba(7,16,29,.35);padding:0;overflow:hidden}.sector-rank-row summary{display:grid;grid-template-columns:46px minmax(170px,1.1fr) minmax(120px,1.6fr) 52px 52px minmax(135px,1fr);gap:10px;align-items:center;padding:11px 12px;border:0;list-style:none}.sector-rank-row summary::-webkit-details-marker{display:none}.sector-rank-row[open]{border-color:#334d70;background:rgba(12,26,44,.65)}.sector-rank{font-weight:800;font-variant-numeric:tabular-nums;font-size:13px;color:#9fb1cb}.sector-rank.rank-top{color:var(--green)}.sector-rank.rank-mid{color:var(--cyan)}.sector-rank.rank-low{color:#778aa6}.sector-title{font-weight:700;font-size:12px;min-width:0}.sector-title small,.sector-leader small{display:block;color:var(--muted);font-size:10px;font-weight:500;margin-top:2px}.sector-meter{height:7px;background:#15243a;border-radius:999px;overflow:hidden}.sector-meter-fill{display:block;height:100%;background:linear-gradient(90deg,#356b92,var(--cyan));border-radius:999px}.sector-strength,.sector-acc{font-size:12px;text-align:right;font-variant-numeric:tabular-nums}.sector-acc.pos{color:var(--green)}.sector-acc.neg{color:var(--rose)}.sector-leader{text-align:left;font-size:12px;color:#eef6ff}.sector-stocks{border-top:1px solid rgba(32,49,76,.6);padding:10px 12px 12px 58px}.sector-stocks-label{font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:.08em;margin-bottom:7px}.stock-chips{display:flex;flex-wrap:wrap;gap:7px}.stock-chip{display:grid;grid-template-columns:auto auto auto;gap:5px;align-items:center;border:1px solid #2a3d5c;border-radius:8px;background:#0b1728;padding:6px 8px;font-size:11px}.stock-chip.best{border-color:#2f7860;background:rgba(69,212,131,.08)}.chip-rank{color:#8094b0}.chip-score{color:var(--cyan);font-variant-numeric:tabular-nums}
    .stage-stack{display:flex;height:20px;background:#152238;border-radius:7px;overflow:hidden;margin:8px 0 14px}.stage-seg{height:100%}.st-leader{background:var(--green)}.st-mature{background:var(--cyan)}.st-emerging{background:var(--amber)}.st-neutral{background:#64748b}.st-fading{background:var(--rose)}.stage-legend{display:grid;grid-template-columns:repeat(2,1fr);gap:8px}.stage-item{display:grid;grid-template-columns:10px 1fr auto auto;gap:7px;align-items:center;font-size:11px;color:#c1cee1}.stage-dot{width:8px;height:8px;border-radius:50%}.stage-item small{color:var(--muted);min-width:28px;text-align:right}
    table.data{width:100%;border-collapse:collapse;font-size:12px}table.data th,table.data td{padding:8px 8px;border-bottom:1px solid rgba(32,49,76,.66);text-align:right;white-space:nowrap}table.data th:first-child,table.data td:first-child,table.data th:nth-child(2),table.data td:nth-child(2){text-align:left}table.data th{color:#91a5c2;font-weight:650;background:rgba(255,255,255,.014);position:sticky;top:0}.table-wrap{overflow:auto;max-height:515px}
    .mini-cards{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}.mini-card{padding:13px}.mini-title{font-size:13px;font-weight:750;margin-bottom:9px}.mini-grid{display:grid;grid-template-columns:1fr auto;gap:5px 10px;font-size:11px;color:var(--muted)}.mini-grid b{color:#e5eefc;font-variant-numeric:tabular-nums}
    .readout{margin:0;padding-left:18px;color:#c8d5e8;font-size:13px;line-height:1.65}.readout b{color:white}.disclaimer{margin-top:20px;padding:14px 16px;border:1px solid #2d3d58;background:#0a1423;border-radius:11px;font-size:11px;color:#8da0bc;line-height:1.55}
    details{margin-top:12px;border-top:1px solid var(--line);padding-top:12px}summary{cursor:pointer;color:#aebdd2;font-size:12px}.method{font-size:12px;color:var(--muted);line-height:1.6}
    @media(max-width:1150px){.kpis{grid-template-columns:repeat(3,1fr)}.grid-2,.grid-even{grid-template-columns:1fr}.grid-3{grid-template-columns:1fr 1fr}}
    @media(max-width:680px){.shell{padding:17px 13px 35px}.topline{align-items:flex-start;flex-direction:column}.kpis{grid-template-columns:1fr 1fr}.grid-3,.mini-cards{grid-template-columns:1fr}.bar-row{grid-template-columns:110px 1fr 38px 42px}.nav{position:static}.big{font-size:23px}.sector-rank-row summary{grid-template-columns:38px 1fr 45px 45px}.sector-meter,.sector-leader{display:none}.sector-stocks{padding-left:12px}}
    """

    market_chart=_line_svg(
        rh if rh is not None else pd.DataFrame(),
        [('market_score','Điểm thị trường','cyan'),('breadth_ma20_pct','Breadth MA20','green'),('breadth_ma50_pct','Breadth MA50','blue')],
        days=60,
    )
    sector_rotation=_rotation_svg(sectors)
    sector_history_chart=_sector_history_svg(sector_history,top_sectors)
    sector_rank_leaders=_sector_rank_leaders(latest,sectors,stocks_per_sector=5)
    scatter=_scatter_svg(latest)
    hist=_histogram_svg(latest['leadership_score'])
    stage_chart=_stage_distribution(latest)
    readout=_signal_readout(latest,rh,sectors)
    bt_cards=_backtest_cards(bt.get('summary'))
    bt_decile=_backtest_decile_svg(bt.get('deciles'),horizon=20)
    bt_ic=_ic_svg(bt.get('ic_daily'),horizon=20)

    html_doc=f"""<!doctype html><html lang='vi'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><meta name='color-scheme' content='dark'><title>Vietnam Market Flow — Research Dashboard</title><style>{css}</style></head><body><div class='shell'>
    <div class='topline'><div><div class='eyebrow'>Finsuccess · Market Intelligence</div><h1>Vietnam Market Flow</h1><div class='muted'>Theo dõi trạng thái thị trường, luân chuyển ngành và độ rộng của nhóm cổ phiếu dẫn dắt.</div></div><div class='tag'>Dữ liệu đến {html.escape(dt)}</div></div>
    <div class='nav'><a href='#market'>Thị trường</a><a href='#sectors'>Ngành</a><a href='#stocks'>Cổ phiếu</a><a href='#validation'>Kiểm định</a><a href='#method'>Phương pháp</a></div>

    <div class='kpis' id='market'>
      <div class='card'><div class='card-label'>Trạng thái</div><div class='big'>{html.escape(regime_text)}</div><div class='sub'>Regime tổng hợp</div></div>
      <div class='card'><div class='card-label'>Điểm thị trường</div><div class='big'>{_num(regime_latest.get('market_score'))}</div><div class='sub'>Thang 0–100</div></div>
      <div class='card'><div class='card-label'>Breadth trên MA20</div><div class='big'>{_num(breadth20,1,'%')}</div><div class='sub'>Sức mạnh ngắn hạn</div></div>
      <div class='card'><div class='card-label'>Breadth trên MA50</div><div class='big'>{_num(breadth50,1,'%')}</div><div class='sub'>Độ bền xu hướng</div></div>
      <div class='card'><div class='card-label'>Nhóm dẫn dắt</div><div class='big'>{leader_count}</div><div class='sub'>Leader + Mature</div></div>
      <div class='card'><div class='card-label'>Độ phủ dữ liệu</div><div class='big'>{coverage}/{reference}</div><div class='sub'>{_num(coverage_pct,0,'%')} universe tham chiếu</div></div>
    </div>

    <div class='grid-2 section'>
      <div class='panel'><div class='section-head'><div><div class='section-kicker'>Market pulse</div><h2>Xu hướng regime & độ rộng</h2></div><span class='tag'>60 phiên</span></div>{market_chart}<div class='note'>Điểm thị trường và breadth cùng quy về thang 0–100 để quan sát hướng đi và phân kỳ.</div></div>
      <div class='panel'><div class='section-kicker'>Signal breadth</div><h2>Cấu trúc tín hiệu hiện tại</h2>{stage_chart}<h3>Điểm đáng chú ý</h3>{readout}</div>
    </div>

    <div class='section' id='sectors'><div class='section-head'><div><div class='section-kicker'>Sector rotation</div><h2>Luân chuyển ngành</h2></div><span class='tag'>Strength × Acceleration</span></div></div>
    <div class='grid-2'>
      <div class='panel'>{sector_rotation}<div class='note'>Trục ngang = sức mạnh ngành; trục dọc = thay đổi điểm trong 5 phiên; kích thước = số mã đủ điều kiện.</div></div>
      <div class='panel'><h2>Xếp hạng sức mạnh ngành</h2>{_sector_bars(sectors,12)}<div class='note'>Cột cuối là mức tăng/giảm điểm sức mạnh so với 5 phiên trước.</div></div>
    </div>
    <div class='panel section'><div class='section-head'><div><div class='section-kicker'>Sector ranking</div><h2>Xếp hạng ngành & cổ phiếu dẫn dắt</h2></div><span class='tag'>Rank theo Sector Score</span></div>
      <div class='note' style='margin-bottom:10px'>Ngành được xếp hạng theo Sector Score giảm dần; trong từng ngành, cổ phiếu được xếp theo Leadership Score. Bấm vào từng ngành để xem top 5 cổ phiếu.</div>
      {sector_rank_leaders}
    </div>
    <div class='panel section'><div class='section-head'><div><div class='section-kicker'>Sector trend</div><h2>Top ngành qua thời gian</h2></div><span class='tag'>40 phiên</span></div>{sector_history_chart}</div>

    <div class='section' id='stocks'><div class='section-head'><div><div class='section-kicker'>Leadership map</div><h2>Cấu trúc cổ phiếu dẫn dắt</h2></div></div></div>
    <div class='grid-2'>
      <div class='panel'><h2>Điểm dẫn dắt × tăng tốc</h2>{scatter}<div class='note'>Kích thước điểm phản ánh Flow Score. Nhãn ưu tiên các mã có điểm/tăng tốc nổi bật.</div></div>
      <div class='panel'><h2>Phân phối điểm dẫn dắt</h2>{hist}<div class='note'>Giúp phân biệt thị trường có leadership lan tỏa hay chỉ tập trung ở một nhóm nhỏ.</div></div>
    </div>

    <div class='grid-even section'>
      <div class='panel'><h2>Cổ phiếu dẫn dắt</h2><div class='table-wrap'>{_table(leaders,['ticker','sector','leadership_score','acceleration','rs_score','flow_score','trend_score','stage'],25)}</div></div>
      <div class='panel'><h2>Đang tăng tốc</h2><div class='table-wrap'>{_table(emerging,['ticker','sector','leadership_score','acceleration','flow_score','sector_score','stage'],25)}</div></div>
    </div>
    <div class='panel section'><h2>Suy yếu / mất động lượng</h2><div class='table-wrap'>{_table(fading,['ticker','sector','leadership_score','acceleration','rs_score','flow_score','stage'],25)}</div></div>

    <div class='section' id='validation'><div class='section-head'><div><div class='section-kicker'>Model validation</div><h2>Kiểm định tín hiệu</h2></div><span class='tag'>Preliminary</span></div></div>
    <div class='mini-cards'>{bt_cards}</div>
    <div class='grid-even section'>
      <div class='panel'><h2>Alpha theo decile — 20 phiên</h2>{bt_decile}<div class='note'>So sánh market-adjusted return giữa các nhóm điểm Leadership. D10 là nhóm điểm cao nhất.</div></div>
      <div class='panel'><h2>Spearman IC — 20 phiên</h2>{bt_ic}<div class='note'>IC dương cho thấy thứ hạng Leadership có quan hệ cùng chiều với forward market alpha.</div></div>
    </div>

    <div class='disclaimer'>Dashboard là công cụ nghiên cứu định lượng, không phải tín hiệu mua/bán tự động. “Dòng tiền” trong V1 là proxy từ giá, khối lượng và giá trị giao dịch, không phải số liệu mua ròng của tổ chức. Backtest hiện dùng universe sản xuất hiện tại nên chưa loại bỏ hoàn toàn survivorship bias; kết quả kiểm định nên được xem là diagnostic cho đến khi hoàn thành point-in-time universe và backfill mã hủy niêm yết.</div>
    <details id='method'><summary>Phương pháp & cách đọc dashboard</summary><div class='method'><p><b>Leadership Score</b> tổng hợp Relative Strength, Flow proxy, Trend quality và Sector confirmation. <b>Acceleration</b> là thay đổi điểm trong 5 phiên. <b>Emerging</b> yêu cầu điểm nền đủ cao và tăng tốc mạnh; <b>Fading</b> phản ánh giảm động lượng hoặc tụt dưới ngưỡng điểm.</p><p>Sector Rotation được dùng để phân biệt cổ phiếu mạnh nhờ riêng lẻ với cổ phiếu được xác nhận bởi ngành. Regime tổng hợp market breadth, return và liquidity để mô tả bối cảnh, không dùng như dự báo chắc chắn cho VN-Index.</p></div></details>
    </div></body></html>"""
    Path(out_path).write_text(html_doc,encoding='utf-8')
