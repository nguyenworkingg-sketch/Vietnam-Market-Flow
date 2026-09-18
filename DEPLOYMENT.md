# Production deployment

## GitHub

The repository runs three layers:

1. Unit tests.
2. Push-time live smoke test on 10 large/liquid tickers.
3. Full production scan at 16:25 Vietnam time on weekdays or via `workflow_dispatch`.

The full scan writes `outputs/` and `docs/index.html` back to `main`. Output-only commits are ignored by the push trigger, preventing recursive workflow loops.

## GitHub Actions secrets

Repository → **Settings → Secrets and variables → Actions → New repository secret**.

Recommended:

- `VNSTOCK_API_KEY` — obtain from your own Vnstock account.
- `SUPABASE_URL` — project URL.
- `SUPABASE_SERVICE_ROLE_KEY` — server-side service-role key. Never put this value in source code, issues, chat, or a public file.

The workflow runs without Supabase secrets; it simply skips DB persistence. Vnstock Guest mode can also run without a Vnstock key, subject to its lower request limit.

## GitHub Pages

After `docs/index.html` exists from a successful full run:

Repository → **Settings → Pages → Deploy from a branch → `main` / `/docs`**.

Expected URL:

`https://nguyenworkingg-sketch.github.io/Vietnam-Market-Flow/`

## Supabase

The project already expects these RLS-enabled tables:

`mf_universe`, `mf_scores_daily`, `mf_sector_daily`, `mf_market_regime`, `mf_runs`.

Daily writes are **latest-only**. Historical backfills should be run explicitly rather than rewriting the full history every afternoon.

## Data-validation gate before trusting signals

Do not treat live scores as portfolio signals until all of these pass:

- Price unit check on at least 20 tickers across HOSE/HNX/UPCOM.
- Trading-value reconciliation against a second public reference on several sessions.
- Corporate-action checks on known cash dividend / stock dividend / rights / split cases.
- Missing-data and suspended-stock handling.
- Sector-mapping sanity check.
- Forward-return backtest with point-in-time liquidity filters and out-of-sample period.
