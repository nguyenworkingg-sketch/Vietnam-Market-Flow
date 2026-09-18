# Vietnam Market Flow Engine

Independent quantitative system for Vietnam equity **market regime, sector rotation, stock leadership and momentum acceleration**. It does **not** use SStock scores or signals.

## What the engine measures

Four transparent cross-sectional scores (0–100):

- **Relative Strength** — stock vs VNIndex and vs its sector.
- **Flow** — trading-value / volume expansion and positive-volume participation.
- **Trend Quality** — MA structure/slope, 52-week-high proximity, consistency and volatility.
- **Sector Confirmation** — sector relative strength, breadth and liquidity expansion.

Default hypothesis:

`Leadership = 35% RS + 25% Flow + 20% Trend + 20% Sector`

Weights are configurable in `config/model.yaml`; they are starting hypotheses and must be validated by out-of-sample backtests.

## Production design

```text
vnstock / market feed
        ↓
canonical OHLCV + trading value
        ↓
feature engine
        ↓
RS / Flow / Trend / Sector
        ↓
Leadership + 5D acceleration
        ↓
Emerging / Leader / Mature / Fading
        ↓
Supabase history + HTML dashboard
```

The live job intentionally scans an **investable, liquidity-prioritized universe** rather than wasting API calls on every illiquid listing. Historical backtests will use time-varying liquidity filters separately to avoid mixing today's universe with the past.

## Run locally

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python scripts/make_demo_data.py
python scripts/run_daily.py --provider csv --csv-folder sample_data
```

Live smoke test:

```bash
python scripts/run_daily.py --provider vnstock --symbols FPT,VCB,GAS,POW,REE,HPG,VNM,SSI,MWG,VIC
```

Full live scan:

```bash
python scripts/run_daily.py --provider vnstock
```

## Data conventions

- Equity prices are normalized to **thousand VND/share** inside the provider adapter.
- Trading value is normalized to **VND**.
- `min_avg_value_20` is therefore a real VND/day liquidity threshold.
- The model uses OHLCV/value as a **market-flow proxy**, not exact institutional net flow.
- Corporate-action behavior from the selected source must be validated before long-horizon production backtests.

## Automation

`.github/workflows/daily.yml` does two different jobs:

- **Push:** tests + a 10-stock live smoke test. This protects the repo from committing broken production output.
- **Weekdays 16:25 Vietnam time / manual dispatch:** full live scan, optional Supabase sync, then dashboard/output commit.

Optional GitHub Actions secrets:

- `VNSTOCK_API_KEY` — recommended Community key for a higher documented request limit than Guest mode.
- `SUPABASE_URL`
- `SUPABASE_SERVICE_ROLE_KEY`

No secrets are required for the CSV demo. The live Vnstock job can run in Guest mode, but more slowly.

## Outputs

- `docs/index.html` — GitHub Pages dashboard after the first full successful production run.
- `outputs/dashboard.html`
- `outputs/scores_latest.csv`
- `outputs/market_regime.csv`
- `outputs/scores_history.parquet` — rolling computation window, not the long-term source of truth.
- Supabase `mf_*` tables — long-term production history when configured.

## Database tables

Production schema uses:

- `mf_universe`
- `mf_scores_daily`
- `mf_sector_daily`
- `mf_market_regime`
- `mf_runs`

## Next quantitative milestones

1. Validate live data units and corporate actions.
2. Historical backfill with delisted / changing-universe handling.
3. Decile event study for +5D/+20D/+60D market- and sector-adjusted returns.
4. Walk-forward / out-of-sample optimization of weights and thresholds.
5. Add foreign, proprietary, ETF and order-flow features only after the base model earns its keep.

## Data-source note

The default live adapter uses the `vnstock` Python package as a client-side connector to third-party market sources. Use it within its licence and the underlying source terms; this repository stores analysis code, not a redistributed market-data service.
