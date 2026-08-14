-- Phase 1 schema: quant_trader
-- Design principles: no silent overwrites, no silent conflict resolution,
-- missing/unresolved data stays visible, everything traceable to a source.

PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS schema_meta (
    schema_version   INTEGER NOT NULL,
    applied_at       TEXT NOT NULL
);

-- ============================================================
-- SECURITY MASTER
-- ============================================================
-- One row per distinct company/security, keyed by a stable internal id.
-- Tickers are NOT the primary key because they are not stable identifiers.
CREATE TABLE IF NOT EXISTS securities (
    security_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    primary_ticker       TEXT NOT NULL,          -- most recent/current known ticker
    name                  TEXT,
    exchange              TEXT,
    country               TEXT,
    currency              TEXT,
    sector                TEXT,
    industry              TEXT,
    asset_type            TEXT NOT NULL DEFAULT 'STOCK',
    isin                  TEXT,
    t212_ticker           TEXT,                  -- populated only if resolvable, not required
    first_seen_date       TEXT,                  -- earliest date this security is known to us
    last_seen_date        TEXT,                  -- latest date this security is known to us
    active_flag           INTEGER NOT NULL DEFAULT 1,
    delisted_date         TEXT,
    delisting_reason       TEXT,                  -- 'acquired'|'bankrupt'|'merged'|'renamed'|'went_private'|'unknown'|NULL
    delisting_confidence    TEXT NOT NULL DEFAULT 'unverified',  -- 'verified'|'unverified'
    delisting_source         TEXT,                  -- URL or citation, where available
    identifier_quality     TEXT NOT NULL DEFAULT 'unresolved',  -- 'resolved'|'partial'|'unresolved'
    cik                    TEXT,                  -- SEC Central Index Key, where resolved
    has_unsupported_corporate_action INTEGER NOT NULL DEFAULT 0,  -- spinoff/rights issue etc: flagged, not processed
    unsupported_corporate_action_note TEXT,
    notes                  TEXT,
    created_at             TEXT NOT NULL,
    updated_at             TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_securities_ticker_unique ON securities(primary_ticker);

-- ============================================================
-- TICKER HISTORY
-- ============================================================
-- Maps every ticker a security has ever used to a validity window,
-- so historical membership rows (which are ticker-keyed) can be
-- resolved to a stable security_id.
CREATE TABLE IF NOT EXISTS ticker_history (
    ticker_history_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    security_id           INTEGER NOT NULL REFERENCES securities(security_id),
    ticker                 TEXT NOT NULL,
    valid_from             TEXT NOT NULL,
    valid_to                TEXT,                 -- NULL = still valid
    source                  TEXT,
    UNIQUE(ticker, valid_from)
);
CREATE INDEX IF NOT EXISTS idx_ticker_history_ticker ON ticker_history(ticker);
CREATE INDEX IF NOT EXISTS idx_ticker_history_security ON ticker_history(security_id);

-- ============================================================
-- DATA SOURCES (registry, incl. tier)
-- ============================================================
CREATE TABLE IF NOT EXISTS data_sources (
    source_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    source_name     TEXT NOT NULL UNIQUE,
    tier             TEXT NOT NULL CHECK(tier IN ('A','B','C')),
    description      TEXT,
    url              TEXT
);

-- ============================================================
-- INDEX MEMBERSHIP  (point-in-time reconstructable)
-- ============================================================
-- A raw, append-only log of what each source CLAIMS about membership.
-- Conflicting claims from different sources are NOT merged/resolved here;
-- both rows persist, and 'verification_status' records the outcome.
CREATE TABLE IF NOT EXISTS index_membership (
    membership_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    security_id           INTEGER REFERENCES securities(security_id),  -- NULL if unresolved
    raw_ticker             TEXT NOT NULL,          -- ticker string as it appeared in the source
    index_name             TEXT NOT NULL,          -- 'SP500' | 'SP400'
    effective_date         TEXT NOT NULL,
    removal_date            TEXT,                   -- NULL = still a member per this source
    announcement_date       TEXT,
    source_id               INTEGER NOT NULL REFERENCES data_sources(source_id),
    source_reference         TEXT,                   -- URL / citation where practical
    confidence               TEXT NOT NULL CHECK(confidence IN ('verified','unverified','conflicting')),
    verification_status       TEXT,                   -- free-text note on how it was (or wasn't) verified
    membership_quality        TEXT NOT NULL DEFAULT 'unresolved',  -- 'complete'|'partial'|'unresolved'
    ingested_at                TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_membership_security ON index_membership(security_id);
CREATE INDEX IF NOT EXISTS idx_membership_index_dates ON index_membership(index_name, effective_date, removal_date);
CREATE INDEX IF NOT EXISTS idx_membership_raw_ticker ON index_membership(raw_ticker);
-- Deliberately NOT unique on (security_id, index_name, effective_date):
-- conflicting sources must be able to coexist as separate rows.

-- ============================================================
-- PRICES
-- ============================================================
CREATE TABLE IF NOT EXISTS prices (
    price_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    security_id         INTEGER NOT NULL REFERENCES securities(security_id),
    date                  TEXT NOT NULL,
    open                  REAL,
    high                  REAL,
    low                   REAL,
    close                 REAL,
    volume                REAL,
    adj_type              TEXT NOT NULL CHECK(adj_type IN ('raw','split_adjusted','total_return')),
    source_id             INTEGER NOT NULL REFERENCES data_sources(source_id),
    price_data_quality     TEXT NOT NULL DEFAULT 'ok',  -- 'ok'|'flagged'|'suspicious'
    ingested_at            TEXT NOT NULL,
    UNIQUE(security_id, date, adj_type, source_id)
);
CREATE INDEX IF NOT EXISTS idx_prices_security_date ON prices(security_id, date);

-- Phase 3: the backtest engine's per-position price lookups filter on
-- (security_id, adj_type, date <= X) for every position at every
-- rebalance date, so this composite index avoids scanning all adj_type
-- rows for the security.
CREATE INDEX IF NOT EXISTS idx_prices_security_adjtype_date ON prices(security_id, adj_type, date);

-- ============================================================
-- CORPORATE ACTIONS
-- ============================================================
CREATE TABLE IF NOT EXISTS corporate_actions (
    action_id            INTEGER PRIMARY KEY AUTOINCREMENT,
    security_id            INTEGER NOT NULL REFERENCES securities(security_id),
    action_type             TEXT NOT NULL,   -- 'split'|'reverse_split'|'dividend'|'special_dividend'|'ticker_change'|'merger'|'acquisition'|'spinoff'
    ex_date                  TEXT NOT NULL,
    ratio_or_value            REAL,
    detail                    TEXT,
    source_id                 INTEGER NOT NULL REFERENCES data_sources(source_id),
    corporate_action_quality   TEXT NOT NULL DEFAULT 'unresolved',  -- 'verified'|'unverified'|'unresolved'
    ingested_at                 TEXT NOT NULL,
    UNIQUE(security_id, action_type, ex_date)
);

-- ============================================================
-- KNOWN TICKER RENAMES (manually curated registry)
-- ============================================================
-- There is no reliable free bulk source that maps "old ticker -> new
-- ticker" for corporate renames. This table is intentionally a small,
-- explicit, append-only registry rather than an attempt at automated
-- detection. Every row must be traceable to a real, checkable event.
CREATE TABLE IF NOT EXISTS known_ticker_renames (
    rename_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    old_ticker        TEXT NOT NULL,
    new_ticker         TEXT NOT NULL,
    effective_date     TEXT NOT NULL,
    source              TEXT,
    confidence           TEXT NOT NULL DEFAULT 'unverified',  -- 'verified'|'unverified'
    notes                TEXT,
    UNIQUE(old_ticker, new_ticker, effective_date)
);

-- ============================================================
-- KNOWN IDENTIFIERS (CIK/ISIN registry -- the authoritative identity layer)
-- ============================================================
-- Maps a ticker STRING, valid over a date window, to a stable CIK/ISIN.
-- This is what identity resolution should prefer over the ticker itself.
-- Same pattern as known_ticker_renames: small, explicit, sourced --
-- never an attempt at automated bulk identity resolution.
CREATE TABLE IF NOT EXISTS known_identifiers (
    identifier_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker              TEXT NOT NULL,
    cik                  TEXT,
    isin                 TEXT,
    valid_from           TEXT NOT NULL,
    valid_to              TEXT,
    source                TEXT,
    confidence             TEXT NOT NULL DEFAULT 'unverified',
    notes                  TEXT
);
CREATE INDEX IF NOT EXISTS idx_known_identifiers_ticker ON known_identifiers(ticker);

-- ============================================================
-- IDENTITY REVIEW QUEUE
-- ============================================================
-- When a ticker cannot be resolved to a CIK/ISIN AND shows a suspicious
-- gap pattern (could be one continuous company, could be ticker reuse by
-- an unrelated company), it is flagged HERE rather than silently merged
-- or silently split. A human resolves it by adding a known_identifiers
-- row (or confirming it's the same entity).
CREATE TABLE IF NOT EXISTS identity_review_queue (
    review_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker              TEXT NOT NULL,
    flag_reason           TEXT NOT NULL,
    period_1_start         TEXT,
    period_1_end            TEXT,
    period_2_start           TEXT,
    period_2_end              TEXT,
    gap_days                   INTEGER,
    resolved                    INTEGER NOT NULL DEFAULT 0,
    resolution_note               TEXT,
    flagged_at                     TEXT NOT NULL
);

-- ============================================================
-- INGESTION ATTEMPTS (per-ticker, per-attempt status taxonomy)
-- ============================================================
-- Per-ticker granularity, distinct from ingestion_log (a per-run
-- summary). "Zero rows returned" isn't one thing -- it can mean a
-- genuinely delisted security (PERMANENT_FAILURE, a delisted-shaped
-- error) or a currently-fine security that the provider silently
-- returned nothing for with no error at all (SUCCESS_EMPTY_PROVIDER).
-- Distinguishing these used to require a manual ticker-by-ticker audit;
-- this table makes it automatic.
CREATE TABLE IF NOT EXISTS ingestion_attempts (
    attempt_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker             TEXT NOT NULL,
    security_id          INTEGER REFERENCES securities(security_id),
    provider               TEXT NOT NULL,
    requested_start          TEXT NOT NULL,
    requested_end              TEXT NOT NULL,
    attempts                    INTEGER NOT NULL DEFAULT 1,
    status                        TEXT NOT NULL CHECK(status IN
        ('SUCCESS_WITH_DATA','SUCCESS_EMPTY_PROVIDER','TRANSIENT_FAILURE','PERMANENT_FAILURE')),
    rows_returned                  INTEGER,
    error_detail                    TEXT,
    attempted_at                     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_ingestion_attempts_ticker ON ingestion_attempts(ticker);
CREATE INDEX IF NOT EXISTS idx_ingestion_attempts_status ON ingestion_attempts(status);

-- ============================================================
-- INGESTION LOG (reproducibility)
-- ============================================================
CREATE TABLE IF NOT EXISTS ingestion_log (
    run_id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    run_timestamp            TEXT NOT NULL,
    run_type                 TEXT NOT NULL,   -- 'membership'|'prices'|'corporate_actions'
    source_id                 INTEGER REFERENCES data_sources(source_id),
    securities_requested       INTEGER,
    securities_succeeded       INTEGER,
    securities_failed          INTEGER,
    rows_inserted               INTEGER,
    rows_skipped_duplicate      INTEGER,
    warnings                    TEXT,
    schema_version               INTEGER
);

-- ============================================================
-- BACKTEST RUNS (Phase 3: reproducibility)
-- ============================================================
-- One row per backtest execution. config_json captures everything needed
-- to reproduce the run byte-for-byte: quality policy, filters, cost
-- assumptions, execution assumptions, universe definition. Two runs with
-- identical config_json + random_seed should produce identical
-- backtest_results rows.
CREATE TABLE IF NOT EXISTS backtest_runs (
    run_id                     INTEGER PRIMARY KEY AUTOINCREMENT,
    run_label                    TEXT,
    created_at                     TEXT NOT NULL,
    config_json                      TEXT NOT NULL,
    random_seed                        INTEGER,
    code_version                         TEXT,
    start_date                             TEXT NOT NULL,
    end_date                                 TEXT NOT NULL,
    rebalance_frequency                        TEXT NOT NULL CHECK(rebalance_frequency IN ('daily','weekly','monthly')),
    universe_definition                          TEXT NOT NULL,
    data_quality_policy_name                       TEXT NOT NULL,
    strategy_name                                    TEXT NOT NULL,
    cost_assumptions_json                              TEXT NOT NULL,
    execution_assumptions_json                           TEXT NOT NULL
);

-- ============================================================
-- BACKTEST RESULTS (per-run performance metrics)
-- ============================================================
CREATE TABLE IF NOT EXISTS backtest_results (
    result_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id             INTEGER NOT NULL REFERENCES backtest_runs(run_id),
    metric_name           TEXT NOT NULL,
    metric_value            REAL,
    period                     TEXT,   -- NULL for whole-run metrics, else e.g. a year for annual returns
    UNIQUE(run_id, metric_name, period)
);

-- ============================================================
-- BACKTEST COVERAGE (per-run, per-rebalance-date survivorship)
-- ============================================================
-- Deliberately NOT derived from the global Stage 3 survivorship figure --
-- every run computes and stores its OWN coverage, since it varies by
-- universe definition, date range, and data-quality policy.
CREATE TABLE IF NOT EXISTS backtest_coverage (
    coverage_id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id                          INTEGER NOT NULL REFERENCES backtest_runs(run_id),
    as_of_date                        TEXT NOT NULL,
    eligible_constituents               INTEGER NOT NULL,
    usable_data_count                     INTEGER NOT NULL,
    excluded_by_quality                     INTEGER NOT NULL,
    provider_empty_count                      INTEGER NOT NULL,
    identity_unresolved_count                   INTEGER NOT NULL,
    partial_history_count                         INTEGER NOT NULL,
    final_tradable_count                            INTEGER NOT NULL,
    UNIQUE(run_id, as_of_date)
);

-- ============================================================
-- BACKTEST POSITIONS (portfolio audit trail)
-- ============================================================
CREATE TABLE IF NOT EXISTS backtest_positions (
    position_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id                INTEGER NOT NULL REFERENCES backtest_runs(run_id),
    as_of_date              TEXT NOT NULL,
    security_id                INTEGER REFERENCES securities(security_id),
    shares                        REAL NOT NULL,
    weight                          REAL NOT NULL,
    position_value                     REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_backtest_positions_run_date ON backtest_positions(run_id, as_of_date);

-- ============================================================
-- ML EXPERIMENTS (Phase 4: reproducibility)
-- ============================================================
-- One row per experiment -- exploratory or confirmatory, baseline or ML.
-- Mirrors backtest_runs' reproducibility guarantee exactly: identical
-- feature_config_json + target_config_json + split_config_json +
-- hyperparameters_json + random_seed must reproduce identical
-- ml_predictions rows and identical downstream backtest_runs results.
-- is_confirmatory + touched_locked_test_set together are what section 17's
-- experiment-tracking control (the confirmatory-experiment cap) is
-- enforced against -- an experiment that touches the locked test set
-- without is_confirmatory=1 already having been frozen beforehand is a
-- process violation, not just a data question.
CREATE TABLE IF NOT EXISTS ml_experiments (
    experiment_id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_label                 TEXT,
    created_at                          TEXT NOT NULL,
    code_version                           TEXT,
    is_confirmatory                           INTEGER NOT NULL DEFAULT 0,   -- 0 = exploratory (train/validation only)
    confirmatory_frozen_at                       TEXT,                        -- when the config was locked, BEFORE touching test data
    touched_locked_test_set                         INTEGER NOT NULL DEFAULT 0,
    robustness_axis                                    TEXT,                    -- NULL for the primary run, else e.g. 'portfolio_size_30'
    universe_policy                                       TEXT NOT NULL,           -- always 'PERMISSIVE' for real experiments; 'STRICT' only for the sanity check
    target_config_json                                       TEXT NOT NULL,           -- benchmark definition, horizon_months, truncated-label handling
    feature_config_json                                         TEXT NOT NULL,           -- exact feature list + params, must match config.yaml phase4.v1_features at experiment time
    split_config_json                                              TEXT NOT NULL,           -- train/validation/test window boundaries, embargo/purge
    model_type                                                        TEXT NOT NULL,           -- 'historical_mean'|'momentum_only'|'ridge'|'logistic'|'elastic_net'|'random_forest'|'gradient_boosting'
    hyperparameters_json                                                 TEXT NOT NULL,
    random_seed                                                             INTEGER,
    portfolio_construction_json                                                TEXT NOT NULL,           -- size, weighting -- see backtest_runs.config_json for the paired backtest run
    cost_config_json                                                           TEXT NOT NULL,           -- must equal config.yaml's backtest.costs at experiment time -- no second cost model
    linked_backtest_run_id                                                        INTEGER REFERENCES backtest_runs(run_id),  -- the Phase 3 engine run this experiment's predictions fed into
    notes                                                                             TEXT
);
CREATE INDEX IF NOT EXISTS idx_ml_experiments_confirmatory ON ml_experiments(is_confirmatory, touched_locked_test_set);

-- ============================================================
-- ML PREDICTIONS (Phase 4: per-security, per-date model output)
-- ============================================================
CREATE TABLE IF NOT EXISTS ml_predictions (
    prediction_id           INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id               INTEGER NOT NULL REFERENCES ml_experiments(experiment_id),
    as_of_date                     TEXT NOT NULL,
    security_id                       INTEGER NOT NULL REFERENCES securities(security_id),
    predicted_value                      REAL NOT NULL,          -- predicted y_i(t,h)
    predicted_rank                          REAL,                    -- cross-sectional percentile rank within ELIG(t) for this date
    realised_label                             REAL,                    -- filled in once t+h has passed and the label resolves; NULL until then
    label_truncated                               INTEGER NOT NULL DEFAULT 0,  -- section 3.4: security delisted/lost coverage before t+h
    UNIQUE(experiment_id, as_of_date, security_id)
);
CREATE INDEX IF NOT EXISTS idx_ml_predictions_experiment_date ON ml_predictions(experiment_id, as_of_date);
