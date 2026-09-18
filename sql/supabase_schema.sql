-- Optional production storage. Safe naming prefix: mf_
-- Run only after choosing a Supabase project.

create table if not exists public.mf_universe (
  ticker text primary key,
  exchange text,
  sector text,
  name text,
  updated_at timestamptz not null default now()
);

create table if not exists public.mf_scores_daily (
  trade_date date not null,
  ticker text not null,
  sector text,
  rs_score double precision,
  flow_score double precision,
  trend_score double precision,
  sector_score double precision,
  leadership_score double precision,
  acceleration double precision,
  stage text,
  created_at timestamptz not null default now(),
  primary key (trade_date, ticker)
);
create index if not exists mf_scores_daily_ticker_date_idx on public.mf_scores_daily(ticker, trade_date desc);
create index if not exists mf_scores_daily_date_score_idx on public.mf_scores_daily(trade_date desc, leadership_score desc);

create table if not exists public.mf_sector_daily (
  trade_date date not null,
  sector text not null,
  sector_score double precision,
  acceleration double precision,
  breadth_ma20 double precision,
  breadth_ma50 double precision,
  created_at timestamptz not null default now(),
  primary key (trade_date, sector)
);

create table if not exists public.mf_market_regime (
  trade_date date primary key,
  market_score double precision,
  breadth_ma20 double precision,
  breadth_ma50 double precision,
  median_ret20 double precision,
  liquidity_ratio double precision,
  regime text,
  coverage_stocks integer,
  coverage_reference integer,
  created_at timestamptz not null default now()
);

create table if not exists public.mf_runs (
  run_id bigint generated always as identity primary key,
  started_at timestamptz not null default now(),
  finished_at timestamptz,
  status text not null,
  provider text,
  latest_trade_date date,
  eligible_stocks integer,
  message text
);

alter table public.mf_universe enable row level security;
alter table public.mf_scores_daily enable row level security;
alter table public.mf_sector_daily enable row level security;
alter table public.mf_market_regime enable row level security;
alter table public.mf_runs enable row level security;
-- Intentionally no anonymous policies. Server-side jobs should use a service-role secret.

alter table public.mf_market_regime add column if not exists coverage_stocks integer;
alter table public.mf_market_regime add column if not exists coverage_reference integer;
