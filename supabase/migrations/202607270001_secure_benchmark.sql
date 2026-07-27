create table if not exists public.user_secrets (
  user_id uuid not null references auth.users(id) on delete cascade,
  secret_name text not null,
  ciphertext text not null,
  updated_at timestamptz not null default now(),
  primary key (user_id, secret_name)
);

create table if not exists public.benchmark_artifacts (
  user_id uuid not null references auth.users(id) on delete cascade,
  run_id text not null,
  case_id text not null,
  cohort_id text not null default '',
  run_status text not null,
  ciphertext text not null,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  primary key (user_id, run_id)
);

alter table public.user_secrets enable row level security;
alter table public.benchmark_artifacts enable row level security;

create policy "users manage own secrets"
on public.user_secrets
for all
using (auth.uid() = user_id)
with check (auth.uid() = user_id);

create policy "users manage own artifacts"
on public.benchmark_artifacts
for all
using (auth.uid() = user_id)
with check (auth.uid() = user_id);

revoke all on public.user_secrets from anon;
revoke all on public.benchmark_artifacts from anon;
grant select, insert, update, delete on public.user_secrets to authenticated;
grant select, insert, update, delete on public.benchmark_artifacts to authenticated;

