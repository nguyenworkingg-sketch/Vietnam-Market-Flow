from __future__ import annotations

from pathlib import Path
import html
import json
import math
import numpy as np
import pandas as pd

from .selection import detect_opportunity_entries, build_model_portfolio, build_entry_candidates, build_entry_signal_history


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
    'short_momentum_score': 'SM ngắn hạn',
    'long_momentum_score': 'SM dài hạn',
    'prev_short_momentum_score': 'SM NH phiên trước',
    'prev_long_momentum_score': 'SM DH phiên trước',
    'portfolio_rank': 'Hạng',
    'sector_rank': 'Hạng ngành',
    'portfolio_score': 'Điểm Port',
    'weight': 'Tỷ trọng %',
    'entry_rank': 'Hạng',
    'entry_score': 'Điểm entry',
    'entry_score_current': 'Điểm entry hiện tại',
    'macd_status': 'MACD',
    'ma_status': 'MA cross',
    'technical_setup': 'Technical setup',
    'entry_date': 'Ngày mở vị thế',
    'entry_price': 'Giá entry',
    'current_price': 'Giá hiện tại',
    'since_entry_pct': 'Từ entry %',
    'entry_age_sessions': 'Số phiên từ entry',
    'entry_reason': 'Trigger',
    'leadership_med10': 'Leadership 10P',
    'long_med10': 'SM dài hạn 10P',
    'sector_med10': 'Sector 10P',
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
        if c == 'since_entry_pct':
            d[c] = d[c].map(lambda v: '' if pd.isna(v) else f'{100*v:+.1f}%')
        else:
            d[c] = d[c].map(lambda v: '' if pd.isna(v) else f'{v:,.1f}')
    for c in [col for col in d.columns if 'date' in str(col).lower()]:
        d[c] = pd.to_datetime(d[c], errors='coerce').map(lambda v: '' if pd.isna(v) else v.strftime('%d/%m/%Y'))
    if 'stage' in d.columns:
        d['stage'] = d['stage'].map(lambda x: STAGE_VI.get(str(x), str(x)))
    d = d.rename(columns={k: v for k, v in DISPLAY.items() if k in d.columns})
    return d.to_html(index=False, border=0, classes='data', escape=True)



def _candlestick_macd_svg(price_history: pd.DataFrame, ticker: str, days: int = 126, width: int = 1400, height: int = 720, entry_events: pd.DataFrame | None = None) -> str:
    if price_history is None or price_history.empty:
        return "<div class='empty'>Chưa có dữ liệu giá cho biểu đồ nến.</div>"
    required = {'date','ticker','open','high','low','close','volume'}
    if not required.issubset(price_history.columns):
        return "<div class='empty'>Thiếu OHLCV để dựng biểu đồ nến.</div>"

    x = price_history[price_history['ticker'].astype(str).eq(str(ticker))].copy()
    if x.empty:
        return "<div class='empty'>Chưa có dữ liệu giá cho mã này.</div>"
    x['date'] = pd.to_datetime(x['date'])
    x = x.sort_values('date').drop_duplicates('date').tail(days).reset_index(drop=True)
    for col in ['open','high','low','close','volume']:
        x[col] = pd.to_numeric(x[col], errors='coerce')
    x = x.dropna(subset=['open','high','low','close'])
    if len(x) < 20:
        return "<div class='empty'>Chưa đủ lịch sử để dựng biểu đồ kỹ thuật.</div>"

    close = x['close']
    x['ma20'] = close.rolling(20, min_periods=20).mean()
    x['ma50'] = close.rolling(50, min_periods=50).mean()
    std20 = close.rolling(20, min_periods=20).std()
    x['bb_upper'] = x['ma20'] + 2.0 * std20
    x['bb_lower'] = x['ma20'] - 2.0 * std20
    ema12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
    ema26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
    x['macd'] = ema12 - ema26
    x['signal'] = x['macd'].ewm(span=9, adjust=False, min_periods=9).mean()
    x['hist'] = x['macd'] - x['signal']

    left, right = 54, 72
    price_top, price_bottom = 28, 445
    vol_top, vol_bottom = 330, 445
    macd_top, macd_bottom = 500, 680
    pw = width - left - right

    price_min = float(np.nanmin([x['low'].min(), x['bb_lower'].min()]))
    price_max = float(np.nanmax([x['high'].max(), x['bb_upper'].max()]))
    if not np.isfinite(price_min) or not np.isfinite(price_max) or price_max <= price_min:
        return "<div class='empty'>Không thể xác định thang giá.</div>"
    pad = (price_max - price_min) * 0.05
    price_min -= pad
    price_max += pad
    max_vol = max(1.0, float(x['volume'].fillna(0).max()))
    macd_vals = pd.concat([x['macd'],x['signal'],x['hist']]).dropna()
    macd_lim = max(1e-9, float(macd_vals.abs().max()) * 1.12) if not macd_vals.empty else 1.0

    n = len(x)
    step = pw / max(1, n)
    candle_w = max(1.4, min(7.0, step * 0.62))

    def sx(i): return left + (i + 0.5) * step
    def sy_price(v): return price_top + (price_max - float(v)) / (price_max - price_min) * (price_bottom - price_top)
    def sy_vol(v): return vol_bottom - float(v) / max_vol * (vol_bottom - vol_top)
    def sy_macd(v): return macd_top + (macd_lim - float(v)) / (2 * macd_lim) * (macd_bottom - macd_top)

    chunks = [
        f"<svg viewBox='0 0 {width} {height}' class='chart-svg candle-chart' role='img' aria-label='Biểu đồ nến {html.escape(str(ticker))}'>",
        f"<rect x='0' y='0' width='{width}' height='{height}' class='terminal-bg'/>",
    ]

    for frac in np.linspace(0,1,6):
        py = price_top + frac*(price_bottom-price_top)
        val = price_max - frac*(price_max-price_min)
        chunks.append(f"<line x1='{left}' y1='{py:.1f}' x2='{width-right}' y2='{py:.1f}' class='terminal-grid'/>")
        chunks.append(f"<text x='{width-right+8}' y='{py+4:.1f}' class='terminal-axis'>{val:,.1f}</text>")
    for frac in np.linspace(0,1,7):
        xx = left + frac*pw
        chunks.append(f"<line x1='{xx:.1f}' y1='{price_top}' x2='{xx:.1f}' y2='{macd_bottom}' class='terminal-grid'/>")

    for col, cls in [('bb_upper','bb-line'),('bb_lower','bb-line'),('ma20','ma20-line'),('ma50','ma50-line')]:
        pts=[]
        for i,row in x.iterrows():
            if pd.notna(row[col]):
                pts.append(f"{sx(i):.1f},{sy_price(row[col]):.1f}")
        if len(pts)>=2:
            chunks.append(f"<polyline points='{' '.join(pts)}' class='{cls}'/>")

    for i,row in x.iterrows():
        up=float(row['close'])>=float(row['open'])
        cls='vol-up' if up else 'vol-down'
        vy=sy_vol(row.get('volume',0) or 0)
        chunks.append(f"<rect x='{sx(i)-candle_w/2:.1f}' y='{vy:.1f}' width='{candle_w:.1f}' height='{max(0.8,vol_bottom-vy):.1f}' class='{cls}'/>")

    for i,row in x.iterrows():
        up=float(row['close'])>=float(row['open'])
        cls='candle-up' if up else 'candle-down'
        xx=sx(i); yhi=sy_price(row['high']); ylo=sy_price(row['low'])
        yo=sy_price(row['open']); yc=sy_price(row['close'])
        body_y=min(yo,yc); body_h=max(1.2,abs(yc-yo))
        date_txt=pd.Timestamp(row['date']).strftime('%d/%m/%Y')
        title=f"{ticker} {date_txt} O:{row['open']:.1f} H:{row['high']:.1f} L:{row['low']:.1f} C:{row['close']:.1f}"
        chunks.append(f"<line x1='{xx:.1f}' y1='{yhi:.1f}' x2='{xx:.1f}' y2='{ylo:.1f}' class='{cls}'><title>{html.escape(title)}</title></line>")
        chunks.append(f"<rect x='{xx-candle_w/2:.1f}' y='{body_y:.1f}' width='{candle_w:.1f}' height='{body_h:.1f}' class='{cls}'><title>{html.escape(title)}</title></rect>")

    last_close=float(x.iloc[-1]['close']); ly=sy_price(last_close)
    chunks.append(f"<line x1='{left}' y1='{ly:.1f}' x2='{width-right}' y2='{ly:.1f}' class='last-price-line'/>")
    chunks.append(f"<rect x='{width-right+2}' y='{ly-10:.1f}' width='64' height='20' rx='4' class='last-price-box'/>")
    chunks.append(f"<text x='{width-right+34}' y='{ly+4:.1f}' text-anchor='middle' class='last-price-text'>{last_close:,.1f}</text>")

    # Mark the dates when the model actually generated a causal entry event.
    if entry_events is not None and not entry_events.empty:
        ev = entry_events[entry_events['ticker'].astype(str).eq(str(ticker))].copy()
        if not ev.empty:
            ev['entry_date'] = pd.to_datetime(ev['entry_date']).dt.normalize()
            date_to_i = {pd.Timestamp(d).normalize(): i for i,d in enumerate(x['date'])}
            visible = ev[ev['entry_date'].isin(date_to_i.keys())].sort_values('entry_date').tail(4)
            for k,(_,erow) in enumerate(visible.iterrows()):
                idx = date_to_i[pd.Timestamp(erow['entry_date']).normalize()]
                ex = sx(idx)
                ep = float(erow['entry_price'])
                ey = sy_price(ep)
                tag_y = price_top + 13 + (k % 2) * 14
                date_txt = pd.Timestamp(erow['entry_date']).strftime('%d/%m')
                chunks.append(f"<line x1='{ex:.1f}' y1='{price_top+28:.1f}' x2='{ex:.1f}' y2='{ey-5:.1f}' class='entry-guide-top'/>")
                chunks.append(f"<polygon points='{ex-5:.1f},{ey-8:.1f} {ex+5:.1f},{ey-8:.1f} {ex:.1f},{ey-1:.1f}' class='entry-arrow'/>")
                chunks.append(f"<text x='{ex:.1f}' y='{tag_y:.1f}' text-anchor='middle' class='entry-top-label'>ENTRY {date_txt}</text>")

            # Compare current price with the most recent model entry.
            last_ev = visible.iloc[-1] if not visible.empty else None
            if last_ev is not None:
                idx = date_to_i[pd.Timestamp(last_ev['entry_date']).normalize()]
                ep = float(last_ev['entry_price'])
                ey = sy_price(ep)
                ex = sx(idx)
                perf = last_close / ep - 1
                chunks.append(f"<line x1='{ex:.1f}' y1='{ey:.1f}' x2='{sx(n-1):.1f}' y2='{ey:.1f}' class='entry-price-line'/>")
                perf_cls = 'entry-perf-pos' if perf >= 0 else 'entry-perf-neg'
                chunks.append(f"<text x='{min(width-right-6,sx(n-1)-4):.1f}' y='{ey-6:.1f}' text-anchor='end' class='{perf_cls}'>Từ entry: {perf*100:+.1f}%</text>")

    chunks.append(f"<line x1='{left}' y1='{macd_top-24}' x2='{width-right}' y2='{macd_top-24}' class='panel-sep'/>")
    zero=sy_macd(0)
    chunks.append(f"<line x1='{left}' y1='{zero:.1f}' x2='{width-right}' y2='{zero:.1f}' class='macd-zero'/>")
    for i,row in x.iterrows():
        hv=row['hist']
        if pd.isna(hv): continue
        yy=sy_macd(hv); cls='macd-bar-pos' if hv>=0 else 'macd-bar-neg'
        chunks.append(f"<rect x='{sx(i)-max(1,candle_w*.42):.1f}' y='{min(yy,zero):.1f}' width='{max(2,candle_w*.84):.1f}' height='{max(1,abs(zero-yy)):.1f}' class='{cls}'/>")
    for col,cls in [('macd','macd-line'),('signal','signal-line')]:
        pts=[]
        for i,row in x.iterrows():
            if pd.notna(row[col]):
                pts.append(f"{sx(i):.1f},{sy_macd(row[col]):.1f}")
        if len(pts)>=2:
            chunks.append(f"<polyline points='{' '.join(pts)}' class='{cls}'/>")

    chunks.append(f"<text x='{left}' y='{macd_top-33}' class='terminal-label'>MACD (12,26,9)</text>")
    chunks.append(f"<text x='{left+118}' y='{macd_top-33}' class='legend-ma20'>MA20</text>")
    chunks.append(f"<text x='{left+164}' y='{macd_top-33}' class='legend-ma50'>MA50</text>")
    chunks.append(f"<text x='{left+210}' y='{macd_top-33}' class='legend-bb'>Bollinger Bands</text>")

    tick_idx=sorted(set(np.linspace(0,n-1,min(7,n)).astype(int).tolist()))
    for i in tick_idx:
        d=pd.Timestamp(x.iloc[i]['date'])
        chunks.append(f"<text x='{sx(i):.1f}' y='{height-16}' text-anchor='middle' class='terminal-axis'>{d.strftime('%m/%Y')}</text>")
    chunks.append("</svg>")
    return ''.join(chunks)


def _candidate_charts(price_history: pd.DataFrame | None, entry_candidates: pd.DataFrame, entry_events: pd.DataFrame | None = None, days: int = 126) -> str:
    if price_history is None or price_history.empty or entry_candidates is None or entry_candidates.empty:
        return "<div class='empty'>Chưa có ứng viên hoặc dữ liệu giá để hiển thị chart.</div>"
    cards=[]
    for i,(_,row) in enumerate(entry_candidates.head(3).iterrows()):
        ticker=str(row['ticker'])
        setup=html.escape(str(row.get('entry_reason',row.get('technical_setup',''))))
        score=_num(row.get('entry_score_current',row.get('entry_score')),1)
        open_attr=" open" if i==0 else ""
        cards.append(
            f"<details class='price-chart-detail'{open_attr}>"
            f"<summary><span>#{i+1} <b>{html.escape(ticker)}</b></span><span class='chart-meta'>Điểm mở vị thế {score} · {setup}</span></summary>"
            f"<div class='terminal-chart-wrap'>{_candlestick_macd_svg(price_history,ticker,days=days,entry_events=entry_events)}</div>"
            "</details>"
        )
    return ''.join(cards)



def _stock_chart_explorer(
    price_history: pd.DataFrame | None,
    entry_events: pd.DataFrame | None,
    latest: pd.DataFrame,
    default_ticker: str | None = None,
    days: int = 126,
) -> str:
    if price_history is None or price_history.empty:
        return "<div class='empty'>Chưa có dữ liệu giá cho Stock Chart Explorer.</div>"

    cols = [
        'date','ticker','open','high','low','close','volume',
        'ma20','ma50','bb_upper','bb_lower','macd','macd_signal','macd_hist',
    ]
    if not {'date','ticker','open','high','low','close','volume'}.issubset(price_history.columns):
        return "<div class='empty'>Thiếu OHLCV cho Stock Chart Explorer.</div>"

    p = price_history[[col for col in cols if col in price_history.columns]].copy()
    p['date'] = pd.to_datetime(p['date'])
    p = (p.sort_values(['ticker','date'])
         .groupby('ticker', group_keys=False)
         .tail(days))
    tickers = sorted(set(latest['ticker'].astype(str)) & set(p['ticker'].astype(str)))
    if not tickers:
        return "<div class='empty'>Không có mã phù hợp để hiển thị chart.</div>"

    def clean(v, digits=4):
        if pd.isna(v):
            return None
        try:
            return round(float(v), digits)
        except Exception:
            return None

    payload = {}
    for ticker, g in p[p['ticker'].astype(str).isin(tickers)].groupby('ticker'):
        rows=[]
        for _,r in g.iterrows():
            rows.append([
                pd.Timestamp(r['date']).strftime('%Y-%m-%d'),
                clean(r.get('open')), clean(r.get('high')), clean(r.get('low')), clean(r.get('close')),
                clean(r.get('volume'),0),
                clean(r.get('ma20')), clean(r.get('ma50')), clean(r.get('bb_upper')), clean(r.get('bb_lower')),
                clean(r.get('macd')), clean(r.get('macd_signal')), clean(r.get('macd_hist')),
            ])
        payload[str(ticker)] = rows

    sig_payload = {ticker: [] for ticker in tickers}
    if entry_events is not None and not entry_events.empty:
        e = entry_events.copy()
        e['entry_date'] = pd.to_datetime(e['entry_date'])
        cutoff = p['date'].min()
        e = e[e['entry_date'].ge(cutoff)]
        for ticker,g in e[e['ticker'].astype(str).isin(tickers)].groupby('ticker'):
            sig_payload[str(ticker)] = [
                [
                    pd.Timestamp(r['entry_date']).strftime('%Y-%m-%d'),
                    clean(r.get('entry_price')),
                    clean(r.get('entry_score')),
                    str(r.get('entry_reason','Entry')),
                ]
                for _,r in g.tail(6).iterrows()
            ]

    current = {}
    for _,r in latest[latest['ticker'].astype(str).isin(tickers)].iterrows():
        current[str(r['ticker'])] = {
            'sector': str(r.get('sector','')),
            'leadership': clean(r.get('leadership_score')),
            'short': clean(r.get('short_momentum_score')),
            'long': clean(r.get('long_momentum_score')),
            'ma20_distance': clean(r.get('ma20_distance')),
            'ret5': clean(r.get('ret_5')),
        }

    if default_ticker not in tickers:
        default_ticker = tickers[0]
    options=''.join(
        f"<option value='{html.escape(t)}'{' selected' if t==default_ticker else ''}>{html.escape(t)}"
        f"{' · '+html.escape(str(current.get(t,{}).get('sector',''))) if current.get(t,{}).get('sector') else ''}</option>"
        for t in tickers
    )
    data_json = json.dumps(payload, ensure_ascii=False, separators=(',',':')).replace('</','<\\/')
    sig_json = json.dumps(sig_payload, ensure_ascii=False, separators=(',',':')).replace('</','<\\/')
    cur_json = json.dumps(current, ensure_ascii=False, separators=(',',':')).replace('</','<\\/')

    return f"""
    <div class='stock-explorer-controls'>
      <label for='stock-chart-select'>Chọn cổ phiếu</label>
      <select id='stock-chart-select'>{options}</select>
      <span id='stock-chart-summary' class='chart-meta'></span>
    </div>
    <div class='canvas-wrap'><canvas id='stock-chart-canvas' height='700'></canvas></div>
    <script type='application/json' id='stock-chart-data'>{data_json}</script>
    <script type='application/json' id='stock-signal-data'>{sig_json}</script>
    <script type='application/json' id='stock-current-data'>{cur_json}</script>
    <script>
    (() => {{
      const DATA=JSON.parse(document.getElementById('stock-chart-data').textContent);
      const SIG=JSON.parse(document.getElementById('stock-signal-data').textContent);
      const CUR=JSON.parse(document.getElementById('stock-current-data').textContent);
      const select=document.getElementById('stock-chart-select');
      const canvas=document.getElementById('stock-chart-canvas');
      const summary=document.getElementById('stock-chart-summary');
      const ctx=canvas.getContext('2d');

      function line(points,color,width=1,dash=[]) {{
        ctx.save(); ctx.strokeStyle=color; ctx.lineWidth=width; ctx.setLineDash(dash);
        ctx.beginPath(); let started=false;
        for(const p of points) {{
          if(p[1]==null || !Number.isFinite(p[1])) {{started=false; continue;}}
          if(!started){{ctx.moveTo(p[0],p[1]);started=true;}} else ctx.lineTo(p[0],p[1]);
        }}
        ctx.stroke(); ctx.restore();
      }}
      function fmt(v,d=1){{return Number.isFinite(v)?v.toFixed(d):'—';}}
      function draw(ticker) {{
        const rows=DATA[ticker]||[]; if(!rows.length)return;
        const dpr=window.devicePixelRatio||1;
        const cssW=Math.max(900,canvas.parentElement.clientWidth-2), cssH=700;
        canvas.style.width=cssW+'px'; canvas.style.height=cssH+'px';
        canvas.width=Math.floor(cssW*dpr); canvas.height=Math.floor(cssH*dpr);
        ctx.setTransform(dpr,0,0,dpr,0,0);
        ctx.clearRect(0,0,cssW,cssH);
        ctx.fillStyle='#07111f';ctx.fillRect(0,0,cssW,cssH);

        const L=54,R=70,PT=24,PB=440,VT=330,VB=440,MT=505,MB=660,W=cssW-L-R;
        const lows=rows.map(r=>r[3]).filter(Number.isFinite), highs=rows.map(r=>r[2]).filter(Number.isFinite);
        const bbl=rows.map(r=>r[9]).filter(Number.isFinite), bbu=rows.map(r=>r[8]).filter(Number.isFinite);
        let pmin=Math.min(...lows,...bbl), pmax=Math.max(...highs,...bbu);
        const pad=(pmax-pmin)*.05||1;pmin-=pad;pmax+=pad;
        const vmax=Math.max(1,...rows.map(r=>r[5]||0));
        const macVals=rows.flatMap(r=>[r[10],r[11],r[12]]).filter(Number.isFinite);
        const mlim=Math.max(.001,...macVals.map(Math.abs))*1.12;
        const step=W/rows.length, cw=Math.max(1.4,Math.min(7,step*.62));
        const sx=i=>L+(i+.5)*step, sy=v=>PT+(pmax-v)/(pmax-pmin)*(PB-PT);
        const vy=v=>VB-(v||0)/vmax*(VB-VT), my=v=>MT+(mlim-v)/(2*mlim)*(MB-MT);

        ctx.strokeStyle='#1a2a40';ctx.lineWidth=1;ctx.fillStyle='#93a6c0';ctx.font='10px system-ui';
        for(let k=0;k<6;k++){{const f=k/5,y=PT+f*(PB-PT),v=pmax-f*(pmax-pmin);ctx.beginPath();ctx.moveTo(L,y);ctx.lineTo(cssW-R,y);ctx.stroke();ctx.fillText(v.toFixed(1),cssW-R+8,y+3);}}
        for(let k=0;k<7;k++){{const x=L+k/6*W;ctx.beginPath();ctx.moveTo(x,PT);ctx.lineTo(x,MB);ctx.stroke();}}

        for(let i=0;i<rows.length;i++) {{
          const r=rows[i], up=r[4]>=r[1], x=sx(i), yv=vy(r[5]);
          ctx.fillStyle=up?'rgba(22,121,111,.75)':'rgba(167,61,75,.72)';
          ctx.fillRect(x-cw/2,yv,cw,Math.max(.8,VB-yv));
        }}
        line(rows.map((r,i)=>[sx(i),r[8]==null?null:sy(r[8])]),'#7186a5',1,[3,4]);
        line(rows.map((r,i)=>[sx(i),r[9]==null?null:sy(r[9])]),'#7186a5',1,[3,4]);
        line(rows.map((r,i)=>[sx(i),r[6]==null?null:sy(r[6])]),'#45d483',1.5);
        line(rows.map((r,i)=>[sx(i),r[7]==null?null:sy(r[7])]),'#f2bf55',1.5);

        for(let i=0;i<rows.length;i++) {{
          const r=rows[i],up=r[4]>=r[1],x=sx(i),color=up?'#18a999':'#ef4d61';
          const yhi=sy(r[2]),ylo=sy(r[3]),yo=sy(r[1]),yc=sy(r[4]);
          ctx.strokeStyle=color;ctx.fillStyle=color;ctx.lineWidth=1;
          ctx.beginPath();ctx.moveTo(x,yhi);ctx.lineTo(x,ylo);ctx.stroke();
          ctx.fillRect(x-cw/2,Math.min(yo,yc),cw,Math.max(1.2,Math.abs(yc-yo)));
        }}

        const zero=my(0);ctx.strokeStyle='#40516d';ctx.beginPath();ctx.moveTo(L,zero);ctx.lineTo(cssW-R,zero);ctx.stroke();
        for(let i=0;i<rows.length;i++) {{
          const h=rows[i][12];if(!Number.isFinite(h))continue;const y=my(h),x=sx(i);
          ctx.fillStyle=h>=0?'#61d4c7':'#f16978';ctx.fillRect(x-cw*.42,Math.min(y,zero),Math.max(2,cw*.84),Math.max(1,Math.abs(zero-y)));
        }}
        line(rows.map((r,i)=>[sx(i),r[10]==null?null:my(r[10])]),'#3da5ff',1.7);
        line(rows.map((r,i)=>[sx(i),r[11]==null?null:my(r[11])]),'#ff8a3d',1.7);

        ctx.fillStyle='#c2cee0';ctx.font='600 11px system-ui';ctx.fillText('MACD (12,26,9)',L,MT-28);
        ctx.fillStyle='#45d483';ctx.fillText('MA20',L+120,MT-28);ctx.fillStyle='#f2bf55';ctx.fillText('MA50',L+166,MT-28);
        ctx.fillStyle='#91a5c2';ctx.fillText('Bollinger Bands',L+212,MT-28);

        const dateIndex=Object.fromEntries(rows.map((r,i)=>[r[0],i]));
        const events=(SIG[ticker]||[]).filter(e=>dateIndex[e[0]]!==undefined);
        events.forEach((e,k)=>{{
          const idx=dateIndex[e[0]],x=sx(idx),y=sy(e[1]);
          const labelY=PT+14+(k%2)*13;
          ctx.strokeStyle='rgba(69,212,131,.55)';ctx.setLineDash([3,5]);
          ctx.beginPath();ctx.moveTo(x,PT+26);ctx.lineTo(x,y-6);ctx.stroke();ctx.setLineDash([]);
          ctx.fillStyle='#45d483';ctx.beginPath();ctx.moveTo(x,y-1);ctx.lineTo(x-5,y-9);ctx.lineTo(x+5,y-9);ctx.closePath();ctx.fill();
          ctx.fillStyle='#9ef0bd';ctx.font='700 9px system-ui';ctx.textAlign='center';
          ctx.fillText('ENTRY '+e[0].slice(8,10)+'/'+e[0].slice(5,7),x,labelY);
          ctx.textAlign='left';
        }});
        if(events.length) {{
          const e=events[events.length-1],idx=dateIndex[e[0]],ep=e[1],y=sy(ep),last=rows[rows.length-1][4],perf=last/ep-1;
          ctx.strokeStyle='#45d483';ctx.setLineDash([6,4]);ctx.beginPath();ctx.moveTo(sx(idx),y);ctx.lineTo(sx(rows.length-1),y);ctx.stroke();ctx.setLineDash([]);
          ctx.fillStyle=perf>=0?'#83e7aa':'#f1919e';ctx.font='700 11px system-ui';ctx.fillText('Từ entry '+(perf*100>=0?'+':'')+(perf*100).toFixed(1)+'%',Math.max(L,sx(rows.length-1)-105),y-7);
        }}

        const ticks=[0,.2,.4,.6,.8,1];
        ctx.fillStyle='#93a6c0';ctx.font='10px system-ui';
        for(const f of ticks){{const i=Math.min(rows.length-1,Math.round(f*(rows.length-1)));ctx.fillText(rows[i][0].slice(5,7)+'/'+rows[i][0].slice(0,4),sx(i)-16,690);}}
        const meta=CUR[ticker]||{{}};
        const ev=events.length?events[events.length-1]:null;
        const last=rows[rows.length-1][4];
        let s=(meta.sector||'')+' · Leadership '+fmt(meta.leadership)+' · SM NH '+fmt(meta.short)+' · SM DH '+fmt(meta.long);
        if(ev) s+=' · Entry '+ev[0]+' @ '+fmt(ev[1])+' · hiện tại '+((last/ev[1]-1)*100>=0?'+':'')+((last/ev[1]-1)*100).toFixed(1)+'%';
        else s+=' · Chưa có entry signal trong 6 tháng';
        summary.textContent=s;
      }}
      select.addEventListener('change',()=>draw(select.value));
      new ResizeObserver(()=>draw(select.value)).observe(canvas.parentElement);
      draw(select.value);
    }})();
    </script>
    """

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


def _line_svg(df: pd.DataFrame, series: list[tuple[str, str, str]], width=920, height=270, days=126) -> str:
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


def _sector_history_svg(sector_history: pd.DataFrame | None, top_sectors: list[str], width=920, height=290, days=126) -> str:
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


def _ic_svg(ic_daily: pd.DataFrame | None, horizon=20, width=620, height=270, days=126) -> str:
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
    opportunity_cfg: dict | None = None,
    price_history: pd.DataFrame | None = None,
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
    opp_cfg=opportunity_cfg or {}
    short_threshold=float(opp_cfg.get('short_threshold',80))
    long_threshold=float(opp_cfg.get('long_threshold',80))
    min_sector_score=float(opp_cfg.get('min_sector_score',50))
    portfolio_size=int(opp_cfg.get('portfolio_size',10))
    portfolio_sector_cap=int(opp_cfg.get('portfolio_sector_cap',2))
    short_entries,long_entries=detect_opportunity_entries(
        scored_history if scored_history is not None else pd.DataFrame(),
        short_threshold=short_threshold,
        long_threshold=long_threshold,
        min_sector_score=min_sector_score,
    )
    model_portfolio=build_model_portfolio(
        latest,
        size=portfolio_size,
        sector_cap=portfolio_sector_cap,
    )
    score_history = scored_history if scored_history is not None else pd.DataFrame()
    entry_kwargs = dict(
        min_sector_score=float(opp_cfg.get('entry_min_sector_score',55)),
        min_leadership_score=float(opp_cfg.get('entry_min_leadership_score',70)),
        min_short_score=float(opp_cfg.get('entry_min_short_score',65)),
        min_long_score=float(opp_cfg.get('entry_min_long_score',70)),
        require_macd_positive=bool(opp_cfg.get('entry_require_macd_positive',True)),
        require_medium_trend=bool(opp_cfg.get('entry_require_medium_trend',True)),
        require_weekly_trend=bool(opp_cfg.get('entry_require_weekly_trend',True)),
        max_ma20_distance=float(opp_cfg.get('entry_max_ma20_distance',0.08)),
        max_ret5=float(opp_cfg.get('entry_max_ret5',0.10)),
        squeeze_bonus=float(opp_cfg.get('entry_squeeze_bonus',5)),
        pullback_bonus=float(opp_cfg.get('entry_pullback_bonus',3)),
    )
    entry_signal_history=build_entry_signal_history(score_history, **entry_kwargs)
    entry_candidates=build_entry_candidates(
        score_history,
        top_n=int(opp_cfg.get('entry_top_n',3)),
        max_age_sessions=int(opp_cfg.get('entry_max_age_sessions',2)),
        max_distance_from_entry=float(opp_cfg.get('entry_max_distance_from_entry',0.05)),
        **entry_kwargs,
    )
    opportunity_count=len(short_entries)+len(long_entries)
    entry_chart_html=_candidate_charts(price_history,entry_candidates,entry_signal_history,days=126)
    default_chart_ticker = (
        str(entry_candidates.iloc[0]['ticker']) if not entry_candidates.empty
        else (str(leaders.iloc[0]['ticker']) if not leaders.empty else None)
    )
    stock_explorer_html=_stock_chart_explorer(
        price_history, entry_signal_history, latest,
        default_ticker=default_chart_ticker, days=126,
    )

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
    .stock-explorer-controls{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:10px}.stock-explorer-controls label{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.08em}.stock-explorer-controls select{min-width:260px;max-width:520px;background:#0a1525;color:#e9f2ff;border:1px solid #2a405f;border-radius:8px;padding:8px 10px;font:inherit}.canvas-wrap{width:100%;overflow:auto;background:#07111f;border:1px solid #20314c;border-radius:10px;padding:6px}.canvas-wrap canvas{display:block;min-width:900px}
    .entry-panel{margin-top:16px;border:1px solid #3b745f;background:linear-gradient(180deg,rgba(19,48,42,.62),rgba(11,26,36,.98));box-shadow:0 16px 42px rgba(0,0,0,.22)}.entry-panel h2{font-size:19px}.entry-rule{display:flex;gap:7px;flex-wrap:wrap;margin:8px 0 12px}.rule-pill{font-size:10px;border:1px solid #315a4d;background:rgba(69,212,131,.07);color:#bdebd0;border-radius:999px;padding:5px 8px}.price-chart-detail{margin-top:10px;border:1px solid #243854;border-radius:10px;background:#081321;overflow:hidden}.price-chart-detail summary{display:flex;justify-content:space-between;gap:12px;align-items:center;padding:10px 12px;border:0;background:#0b1728}.chart-meta{font-size:11px;color:var(--muted);font-weight:500}.terminal-chart-wrap{padding:8px;background:#07111f;overflow:auto}.candle-chart{min-width:980px;background:#07111f;border-radius:8px}.terminal-bg{fill:#07111f}.terminal-grid{stroke:#1a2a40;stroke-width:1;opacity:.72}.terminal-axis,.terminal-label{fill:#93a6c0;font-size:10px}.terminal-label{fill:#c2cee0;font-weight:650}.panel-sep{stroke:#263a55}.candle-up{fill:#18a999;stroke:#18a999;stroke-width:1}.candle-down{fill:#ef4d61;stroke:#ef4d61;stroke-width:1}.vol-up{fill:#16796f;opacity:.75}.vol-down{fill:#a73d4b;opacity:.72}.ma20-line{fill:none;stroke:#45d483;stroke-width:1.6}.ma50-line{fill:none;stroke:#f2bf55;stroke-width:1.5}.bb-line{fill:none;stroke:#7186a5;stroke-width:1;stroke-dasharray:3 4;opacity:.65}.last-price-line{stroke:#20b8a8;stroke-width:1;stroke-dasharray:2 3;opacity:.65}.last-price-box{fill:#148f84}.last-price-text{fill:white;font-size:10px;font-weight:700}.entry-guide{stroke:#45d483;stroke-width:1.2;stroke-dasharray:4 4;opacity:.85}.entry-guide-top{stroke:#45d483;stroke-width:1;stroke-dasharray:3 5;opacity:.55}.entry-arrow{fill:#45d483}.entry-top-label{fill:#9ef0bd;font-size:9px;font-weight:800;letter-spacing:.03em;paint-order:stroke;stroke:#07111f;stroke-width:3px}.entry-price-line{stroke:#45d483;stroke-width:1.2;stroke-dasharray:6 4;opacity:.72}.entry-perf-pos{fill:#83e7aa;font-size:10px;font-weight:750}.entry-perf-neg{fill:#f1919e;font-size:10px;font-weight:750}.macd-zero{stroke:#40516d;stroke-width:1}.macd-bar-pos{fill:#61d4c7;opacity:.9}.macd-bar-neg{fill:#f16978;opacity:.9}.macd-line{fill:none;stroke:#3da5ff;stroke-width:1.7}.signal-line{fill:none;stroke:#ff8a3d;stroke-width:1.7}.legend-ma20{fill:#45d483;font-size:10px}.legend-ma50{fill:#f2bf55;font-size:10px}.legend-bb{fill:#91a5c2;font-size:10px}
    .opportunity-shell{border:1px solid #34506f;background:linear-gradient(180deg,rgba(16,37,57,.98),rgba(10,23,39,.98));box-shadow:0 0 0 1px rgba(84,215,239,.05),0 18px 44px rgba(0,0,0,.18)}.opportunity-shell.has-alert{border-color:#3f856c;box-shadow:0 0 0 1px rgba(69,212,131,.10),0 18px 44px rgba(0,0,0,.20)}.opportunity-title{display:flex;align-items:center;gap:9px}.pulse-dot{width:9px;height:9px;border-radius:50%;background:#66778f}.has-alert .pulse-dot{background:var(--green);box-shadow:0 0 0 5px rgba(69,212,131,.10)}.count-badge{display:inline-flex;align-items:center;justify-content:center;min-width:24px;height:22px;border-radius:999px;padding:0 7px;background:#142943;border:1px solid #2f4c70;color:#dcecff;font-size:11px;font-weight:750}.has-alert .count-badge{background:rgba(69,212,131,.10);border-color:#34745e;color:#8ff0b5}.opp-col{min-width:0}.opp-label{display:flex;align-items:center;justify-content:space-between;gap:8px;margin-bottom:8px;font-size:12px;font-weight:750}.opp-threshold{font-size:10px;color:var(--muted);font-weight:500}.portfolio-note{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin:8px 0 12px}.portfolio-pill{border:1px solid #2b405f;background:#0b1728;border-radius:9px;padding:8px 10px;font-size:11px;color:#aebdd2}.portfolio-pill b{display:block;color:#eef5ff;font-size:13px;margin-top:2px}
    details{margin-top:12px;border-top:1px solid var(--line);padding-top:12px}summary{cursor:pointer;color:#aebdd2;font-size:12px}.method{font-size:12px;color:var(--muted);line-height:1.6}
    @media(max-width:1150px){.kpis{grid-template-columns:repeat(3,1fr)}.grid-2,.grid-even{grid-template-columns:1fr}.grid-3{grid-template-columns:1fr 1fr}}
    @media(max-width:680px){.shell{padding:17px 13px 35px}.topline{align-items:flex-start;flex-direction:column}.kpis{grid-template-columns:1fr 1fr}.grid-3,.mini-cards{grid-template-columns:1fr}.bar-row{grid-template-columns:110px 1fr 38px 42px}.nav{position:static}.big{font-size:23px}.sector-rank-row summary{grid-template-columns:38px 1fr 45px 45px}.sector-meter,.sector-leader{display:none}.sector-stocks{padding-left:12px}}
    """

    market_chart=_line_svg(
        rh if rh is not None else pd.DataFrame(),
        [('market_score','Điểm thị trường','cyan'),('breadth_ma20_pct','Breadth MA20','green'),('breadth_ma50_pct','Breadth MA50','blue')],
        days=126,
    )
    sector_rotation=_rotation_svg(sectors)
    sector_history_chart=_sector_history_svg(sector_history,top_sectors,days=126)
    sector_rank_leaders=_sector_rank_leaders(latest,sectors,stocks_per_sector=5)
    scatter=_scatter_svg(latest)
    hist=_histogram_svg(latest['leadership_score'])
    stage_chart=_stage_distribution(latest)
    readout=_signal_readout(latest,rh,sectors)
    bt_cards=_backtest_cards(bt.get('summary'))
    bt_decile=_backtest_decile_svg(bt.get('deciles'),horizon=20)
    bt_ic=_ic_svg(bt.get('ic_daily'),horizon=20,days=126)

    html_doc=f"""<!doctype html><html lang='vi'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><meta name='color-scheme' content='dark'><title>Vietnam Market Flow — Research Dashboard</title><style>{css}</style></head><body><div class='shell'>
    <div class='topline'><div><div class='eyebrow'>Finsuccess · Market Intelligence</div><h1>Vietnam Market Flow</h1><div class='muted'>Theo dõi trạng thái thị trường, luân chuyển ngành và độ rộng của nhóm cổ phiếu dẫn dắt.</div></div><div class='tag'>Dữ liệu đến {html.escape(dt)}</div></div>

    <div class='panel entry-panel' id='entry-top3'>
      <div class='section-head'><div><div class='section-kicker'>Priority setup</div><h2>Top 3 ứng viên mở vị thế — Model</h2></div><span class='tag'>Strict technical gate</span></div>
      <div class='entry-rule'><span class='rule-pill'>Daily + Weekly trend confirm</span><span class='rule-pill'>RS 60/120 &gt; benchmark</span><span class='rule-pill'>RS 60 &gt; sector</span><span class='rule-pill'>Leadership bền 10 phiên</span><span class='rule-pill'>Entry gần MA20</span><span class='rule-pill'>≤ +10% / 5 phiên</span><span class='rule-pill'>≤ 2 phiên từ entry</span><span class='rule-pill'>≤ +5% từ entry</span></div>
      <div class='note' style='margin-bottom:10px'>V3 chỉ tìm <b>real strength đã được xác nhận ở khung lớn</b>, sau đó chờ pullback-resume hoặc breakout từ vùng siết để vào. MA cross và việc điểm momentum vừa vượt 80 không còn được dùng làm trigger độc lập, vì hai tín hiệu này dễ xuất hiện sau khi giá đã chạy xa.</div>
      <div class='table-wrap'>{_table(entry_candidates,['entry_rank','ticker','sector','entry_date','entry_price','current_price','since_entry_pct','entry_age_sessions','entry_reason','entry_score_current','leadership_med10','long_med10','sector_med10','stage'],3)}</div>
      <div class='section-head' style='margin-top:14px'><div><div class='section-kicker'>6-month technical chart</div><h2>Biểu đồ nến · Volume · MACD</h2></div><span class='tag'>~126 phiên</span></div>
      {entry_chart_html}
    </div>

    <div class='panel section' id='chart-explorer'>
      <div class='section-head'><div><div class='section-kicker'>Stock chart explorer</div><h2>Biểu đồ kỹ thuật toàn bộ cổ phiếu</h2></div><span class='tag'>6 tháng · Entry history</span></div>
      <div class='note' style='margin-bottom:10px'>Chọn bất kỳ mã nào trong universe hiện tại để xem nến, volume, MA20/MA50, Bollinger Bands, MACD và các điểm entry mà model đã phát ra trong lịch sử 6 tháng. Đường entry gần nhất được kéo sang hiện tại để thấy cổ phiếu đã đi xa bao nhiêu.</div>
      {stock_explorer_html}
    </div>

    <div class='nav'><a href='#entry-top3'>Fresh entry</a><a href='#chart-explorer'>Chart cổ phiếu</a><a href='#market'>Thị trường</a><a href='#opportunities'>Cơ hội mới</a><a href='#portfolio'>Model Port 10</a><a href='#sectors'>Ngành</a><a href='#stocks'>Cổ phiếu</a><a href='#validation'>Kiểm định</a><a href='#method'>Phương pháp</a></div>

    <div class='kpis' id='market'>
      <div class='card'><div class='card-label'>Trạng thái</div><div class='big'>{html.escape(regime_text)}</div><div class='sub'>Regime tổng hợp</div></div>
      <div class='card'><div class='card-label'>Điểm thị trường</div><div class='big'>{_num(regime_latest.get('market_score'))}</div><div class='sub'>Thang 0–100</div></div>
      <div class='card'><div class='card-label'>Breadth trên MA20</div><div class='big'>{_num(breadth20,1,'%')}</div><div class='sub'>Sức mạnh ngắn hạn</div></div>
      <div class='card'><div class='card-label'>Breadth trên MA50</div><div class='big'>{_num(breadth50,1,'%')}</div><div class='sub'>Độ bền xu hướng</div></div>
      <div class='card'><div class='card-label'>Nhóm dẫn dắt</div><div class='big'>{leader_count}</div><div class='sub'>Leader + Mature</div></div>
      <div class='card'><div class='card-label'>Độ phủ dữ liệu</div><div class='big'>{coverage}/{reference}</div><div class='sub'>{_num(coverage_pct,0,'%')} universe tham chiếu</div></div>
    </div>

    <div class='grid-2 section'>
      <div class='panel'><div class='section-head'><div><div class='section-kicker'>Market pulse</div><h2>Xu hướng regime & độ rộng</h2></div><span class='tag'>6 tháng · ~126 phiên</span></div>{market_chart}<div class='note'>Điểm thị trường và breadth cùng quy về thang 0–100 để quan sát hướng đi và phân kỳ.</div></div>
      <div class='panel'><div class='section-kicker'>Signal breadth</div><h2>Cấu trúc tín hiệu hiện tại</h2>{stage_chart}<h3>Điểm đáng chú ý</h3>{readout}</div>
    </div>

    <div class='panel section opportunity-shell {'has-alert' if opportunity_count else ''}' id='opportunities'>
      <div class='section-head'><div><div class='section-kicker'>Daily opportunity monitor</div><div class='opportunity-title'><span class='pulse-dot'></span><h2 style='margin:0'>Cơ hội mới vào Top hôm nay</h2><span class='count-badge'>{opportunity_count}</span></div></div><span class='tag'>Cross threshold</span></div>
      <div class='note' style='margin-bottom:12px'>Chỉ hiện mã vừa vượt ngưỡng trong phiên mới nhất và có Sector Score ≥ {min_sector_score:.0f}. Đây là screen độc lập của Vietnam Market Flow, không dùng công thức của nền tảng khác.</div>
      <div class='grid-even'>
        <div class='opp-col'><div class='opp-label'><span>Ngắn hạn</span><span class='opp-threshold'>SM ngắn hạn ≥ {short_threshold:.0f}</span></div><div class='table-wrap'>{_table(short_entries,['ticker','sector','short_momentum_score','prev_short_momentum_score','leadership_score','acceleration','flow_score','sector_score','stage'],15)}</div></div>
        <div class='opp-col'><div class='opp-label'><span>Dài hạn</span><span class='opp-threshold'>SM dài hạn ≥ {long_threshold:.0f}</span></div><div class='table-wrap'>{_table(long_entries,['ticker','sector','long_momentum_score','prev_long_momentum_score','leadership_score','acceleration','trend_score','sector_score','stage'],15)}</div></div>
      </div>
    </div>

    <div class='panel section' id='portfolio'>
      <div class='section-head'><div><div class='section-kicker'>Sector-first model portfolio</div><h2>Model Portfolio — 10 cổ phiếu mạnh</h2></div><span class='tag'>Equal weight</span></div>
      <div class='portfolio-note'><div class='portfolio-pill'>Bước 1<b>Xếp hạng ngành</b></div><div class='portfolio-pill'>Bước 2<b>Chọn mã mạnh trong ngành</b></div><div class='portfolio-pill'>Bước 3<b>Tối đa {portfolio_sector_cap} mã/ngành</b></div></div>
      <div class='note' style='margin-bottom:10px'>Port nghiên cứu ưu tiên ngành mạnh trước, sau đó chọn cổ phiếu có Leadership + SM ngắn hạn + SM dài hạn + Flow tốt nhất. Tỷ trọng mặc định chia đều; không phải danh mục tối ưu hóa rủi ro/lợi nhuận.</div>
      <div class='table-wrap'>{_table(model_portfolio,['portfolio_rank','ticker','sector','sector_rank','sector_score','leadership_score','short_momentum_score','long_momentum_score','portfolio_score','weight','stage'],portfolio_size)}</div>
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
    <div class='panel section'><div class='section-head'><div><div class='section-kicker'>Sector trend</div><h2>Top ngành qua thời gian</h2></div><span class='tag'>6 tháng · ~126 phiên</span></div>{sector_history_chart}</div>

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
    <details id='method'><summary>Phương pháp & cách đọc dashboard</summary><div class='method'><p><b>Leadership Score</b> tổng hợp Relative Strength, Flow proxy, Trend quality và Sector confirmation. <b>Acceleration</b> là thay đổi điểm trong 5 phiên. <b>SM ngắn hạn</b> nhấn mạnh RS 5/20 phiên, thanh khoản, MA20 slope và participation; <b>SM dài hạn</b> nhấn mạnh RS 60/120 phiên, MA50 slope, vị trí so với MA50 và đỉnh 52 tuần. Hai điểm đều là percentile cross-section 0–100.</p><p><b>Fresh Entry V3</b> tách Strength và Timing. Strength phải tồn tại bền qua 10 phiên, vượt benchmark/sector ở 60–120 phiên, đồng thời cấu trúc daily và weekly đều xác nhận. Timing chỉ xuất hiện khi giá quay lại gần MA20 rồi resume, hoặc breakout khỏi vùng Bollinger squeeze nhưng chưa bị kéo xa. <b>Model Portfolio 10</b> vẫn đi theo funnel ngành → cổ phiếu và không đồng nghĩa với điểm mở vị thế mới.</p><p>Sector Rotation được dùng để phân biệt cổ phiếu mạnh nhờ riêng lẻ với cổ phiếu được xác nhận bởi ngành. Regime tổng hợp market breadth, return và liquidity để mô tả bối cảnh, không dùng như dự báo chắc chắn cho VN-Index.</p></div></details>
    </div></body></html>"""
    Path(out_path).write_text(html_doc,encoding='utf-8')
