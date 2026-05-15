-- ============================================================
-- Fatigue Detector — Supabase schema
-- Paste this into: Supabase dashboard → SQL Editor → Run
-- ============================================================

-- 1. Classified runs (per-user history shown in charts)
CREATE TABLE IF NOT EXISTS classified_runs (
  id           BIGSERIAL PRIMARY KEY,
  user_id      UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  run_id       TEXT NOT NULL,
  date         TEXT,
  session_type TEXT,
  neuro_tag    TEXT,
  final_label  TEXT,
  base_label   TEXT,
  final_code   TEXT,
  cml_z        FLOAT,
  gpv_z        FLOAT,
  cml_adj      FLOAT,
  gpv_adj      FLOAT,
  reasons      TEXT,
  created_at   TIMESTAMPTZ DEFAULT NOW(),
  UNIQUE(user_id, run_id)
);
ALTER TABLE classified_runs ENABLE ROW LEVEL SECURITY;
CREATE POLICY "service_full_access" ON classified_runs USING (true);

-- 2. Raw features history (used to compute z-scores per user)
CREATE TABLE IF NOT EXISTS features_history (
  id           BIGSERIAL PRIMARY KEY,
  user_id      UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  run_id       TEXT NOT NULL,
  session_type TEXT,
  neuro_tag    TEXT,
  features     JSONB NOT NULL DEFAULT '{}',
  created_at   TIMESTAMPTZ DEFAULT NOW(),
  UNIQUE(user_id, run_id)
);
ALTER TABLE features_history ENABLE ROW LEVEL SECURITY;
CREATE POLICY "service_full_access" ON features_history USING (true);

-- 3. User integrations (encrypted Garmin / Strava credentials)
CREATE TABLE IF NOT EXISTS user_integrations (
  user_id                  UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
  garmin_email             TEXT,
  garmin_password_enc      TEXT,
  strava_client_id         TEXT,
  strava_client_secret_enc TEXT,
  strava_refresh_token_enc TEXT,
  updated_at               TIMESTAMPTZ DEFAULT NOW()
);
ALTER TABLE user_integrations ENABLE ROW LEVEL SECURITY;
CREATE POLICY "service_full_access" ON user_integrations USING (true);
