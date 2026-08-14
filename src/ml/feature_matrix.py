"""
Assembles the per-(security, as_of_date) feature matrix from
src/ml/features.py's individual point-in-time functions, plus the
matching label from src/ml/targets.py. This is the single place the V1
feature list (config.yaml's phase4.v1_features) gets wired together into
rows a model can train on.

FEATURE_FAMILIES below must stay in sync with config.yaml's
phase4.v1_features -- test_ml_feature_matrix.py asserts this directly so
the two can't silently drift apart.
"""
from src.backtest.universe import build_eligible_universe
from src.ml import features as F
from src.ml import targets as T

FEATURE_FAMILIES = {
    "price_momentum": [
        "return_1m", "return_3m", "return_6m", "return_12m",
        "momentum_acceleration", "distance_from_200d_ma",
    ],
    "volatility_risk": [
        "realised_volatility_63d", "downside_volatility_63d", "max_drawdown_252d",
    ],
    "volume_liquidity": [
        "dollar_volume", "rolling_avg_dollar_volume_20d", "volume_trend_20d_100d",
    ],
    "relative_cross_sectional": [
        "return_relative_to_eligible_mean_63d", "momentum_percentile_rank", "volatility_percentile_rank",
    ],
    "market_regime": [
        "proxy_index_trend", "proxy_index_volatility", "breadth_proxy",
    ],
}

# The cumulative ablation ladder: each step is the union of the families
# up to and including it, not a single family in isolation (PRICE ONLY ->
# PRICE+VOLATILITY -> ... -> FULL).
ABLATION_STEPS = [
    ("price_only", ["price_momentum"]),
    ("price_plus_volatility", ["price_momentum", "volatility_risk"]),
    ("price_plus_volume", ["price_momentum", "volatility_risk", "volume_liquidity"]),
    ("price_plus_market_regime", ["price_momentum", "volatility_risk", "volume_liquidity", "market_regime"]),
    ("full_v1_feature_set", ["price_momentum", "volatility_risk", "volume_liquidity",
                              "relative_cross_sectional", "market_regime"]),
]


def all_v1_feature_names():
    names = []
    for family_features in FEATURE_FAMILIES.values():
        names.extend(family_features)
    return names


def feature_names_for_families(family_names):
    names = []
    for fam in family_names:
        names.extend(FEATURE_FAMILIES[fam])
    return names


def build_feature_matrix_for_date(conn, as_of_date, eligible_ids):
    """Returns {security_id: {feature_name: value_or_None}} for every V1
    feature, for every security in eligible_ids (ELIG(t), passed in and
    never recomputed here). Two passes: per-security "independent"
    features first, then the cross-sectional/market-regime features that
    need the whole eligible set's independent values."""
    per_security = {}
    for sid in eligible_ids:
        per_security[sid] = {
            "return_1m": F.return_1m(conn, sid, as_of_date),
            "return_3m": F.return_3m(conn, sid, as_of_date),
            "return_6m": F.return_6m(conn, sid, as_of_date),
            "return_12m": F.return_12m(conn, sid, as_of_date),
            "momentum_acceleration": F.momentum_acceleration(conn, sid, as_of_date),
            "distance_from_200d_ma": F.distance_from_200d_ma(conn, sid, as_of_date),
            "realised_volatility_63d": F.realised_volatility_63d(conn, sid, as_of_date),
            "downside_volatility_63d": F.downside_volatility_63d(conn, sid, as_of_date),
            "max_drawdown_252d": F.max_drawdown_252d(conn, sid, as_of_date),
            "dollar_volume": F.dollar_volume(conn, sid, as_of_date),
            "rolling_avg_dollar_volume_20d": F.rolling_avg_dollar_volume_20d(conn, sid, as_of_date),
            "volume_trend_20d_100d": F.volume_trend_20d_100d(conn, sid, as_of_date),
        }

    return_3m_map = {sid: v["return_3m"] for sid, v in per_security.items()}
    return_12m_map = {sid: v["return_12m"] for sid, v in per_security.items()}
    vol_map = {sid: v["realised_volatility_63d"] for sid, v in per_security.items()}
    dist_map = {sid: v["distance_from_200d_ma"] for sid, v in per_security.items()}

    proxy_trend = F.proxy_index_trend(return_3m_map)
    proxy_vol = F.proxy_index_volatility(conn, list(eligible_ids), as_of_date, n_sessions=63)
    breadth = F.breadth_proxy(dist_map)

    for sid in eligible_ids:
        per_security[sid]["return_relative_to_eligible_mean_63d"] = F.return_relative_to_eligible_mean(sid, return_3m_map)
        per_security[sid]["momentum_percentile_rank"] = F.cross_sectional_percentile_rank(sid, return_12m_map)
        per_security[sid]["volatility_percentile_rank"] = F.cross_sectional_percentile_rank(sid, vol_map)
        per_security[sid]["proxy_index_trend"] = proxy_trend
        per_security[sid]["proxy_index_volatility"] = proxy_vol
        per_security[sid]["breadth_proxy"] = breadth

    return per_security


def build_panel(conn, cfg, policy_name, as_of_dates, rebalance_dates, horizon_months,
                 predeclared_filters, universe_definition="SP500", universe_cache=None,
                 progress_every=10, log=print):
    """Builds the full panel of rows across as_of_dates (a purged
    train_dates or validation_dates list from walk_forward.build_primary_split).
    For each as_of_date t, resolves target_date = rebalance_dates[index(t) + horizon_months]
    -- horizon_months is an index offset into rebalance_dates, since
    rebalance_dates already has exactly one entry per rebalance period.
    Returns a list of row dicts:
        {"security_id", "as_of_date", "target_date", <feature_name>: value, "y", "rank", "z", "label_truncated"}
    Rows with any missing V1 feature or an unresolvable label are
    excluded and counted, never imputed -- the returned second value is
    {"rows_built", "rows_dropped_missing_feature", "rows_dropped_no_label",
    "excluded_no_price_total", "truncated_total"}.
    """
    policy = cfg["backtest"]["data_quality_policies"][policy_name]
    lookback_days = cfg["backtest"]["execution"]["lookback_days_required"]
    universe_cache = {} if universe_cache is None else universe_cache

    rows = []
    stats = {"rows_built": 0, "rows_dropped_missing_feature": 0, "rows_dropped_no_label": 0,
             "excluded_no_price_total": 0, "truncated_total": 0}

    feature_names = all_v1_feature_names()

    for i, as_of_date in enumerate(as_of_dates):
        idx = rebalance_dates.index(as_of_date)
        target_idx = idx + horizon_months
        if target_idx >= len(rebalance_dates):
            continue  # no resolvable target_date at all -- shouldn't happen given embargo, but never silently proceed
        target_date = rebalance_dates[target_idx]

        eligible, _ = build_eligible_universe(
            conn, as_of_date, policy, predeclared_filters=predeclared_filters,
            universe_definition=universe_definition, lookback_days=lookback_days,
        )
        if not eligible:
            continue

        fmatrix = build_feature_matrix_for_date(conn, as_of_date, eligible)
        labels = T.compute_labels_for_universe(conn, eligible, as_of_date, target_date)
        stats["excluded_no_price_total"] += len(labels["excluded_no_price"])
        stats["truncated_total"] += labels["truncated_count"]

        for sid in eligible:
            if sid not in labels["per_security"]:
                stats["rows_dropped_no_label"] += 1
                continue
            feats = fmatrix[sid]
            if any(feats.get(name) is None for name in feature_names):
                stats["rows_dropped_missing_feature"] += 1
                continue
            label = labels["per_security"][sid]
            row = dict(feats)
            row.update({
                "security_id": sid, "as_of_date": as_of_date, "target_date": target_date,
                "y": label["y"], "rank": label["rank"], "z": label["z"],
                "label_truncated": label["label_truncated"],
            })
            rows.append(row)
            stats["rows_built"] += 1

        if log and (i + 1) % progress_every == 0:
            log(f"    ... {i + 1}/{len(as_of_dates)} dates processed, {stats['rows_built']} rows so far")

    return rows, stats
