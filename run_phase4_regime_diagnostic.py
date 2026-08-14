"""
Regime diagnostic layered on top of the already-completed, exploratory
baselines run.

Narrow question this answers: is the negative validation IC seen across
the linear baselines primarily driven by the Aug-Nov 2020 regime (a
widely documented cross-sectional momentum/quality breakdown during the
COVID recovery rotation), or does the lack of predictive signal persist
outside that window?

This script doesn't retrain, retune, or refit anything. It reads the
per-date IC values that PHASE4_BASELINES_REPORT.json already computed and
persisted and re-aggregates them with/without four specific months
excluded:
  - No feature definitions touched.
  - No model configurations touched -- this uses the exact same
    already-fitted, already-tuned predictions already in ml_predictions
    (experiment_ids unchanged).
  - No hyperparameter retuning of any kind.
  - No access to the locked test period -- this only reads
    PHASE4_BASELINES_REPORT.json, which was built exclusively from
    train_dates/validation_dates.
  - No selection of a "winning" model based on this diagnostic's result;
    it's read-only and answers one narrow question, nothing more.

Run this locally (or anywhere PHASE4_BASELINES_REPORT.json already
exists -- no database access needed):
    python3 run_phase4_regime_diagnostic.py
"""
import json
import os
import statistics

REPORT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "PHASE4_BASELINES_REPORT.json")
OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "PHASE4_EXPLORATORY_VALIDATION_REGIME_DIAGNOSTIC.json")

EXCLUDED_MONTHS = ["2020-08-01", "2020-09-01", "2020-10-01", "2020-11-01"]


def _entries_for(results, path):
    """path is either a top-level key (historical_mean, momentum_only) or a
    (ablation_step, model_type) tuple."""
    if isinstance(path, tuple):
        step, model = path
        return results[step][model]
    return results[path]


def _summarize(per_date, excluded):
    included = {d: v for d, v in per_date.items() if d not in excluded}
    excluded_rows = {d: v for d, v in per_date.items() if d in excluded}
    ics_all = [v["ic"] for v in per_date.values() if v["ic"] is not None]
    ics_incl = [v["ic"] for v in included.values() if v["ic"] is not None]
    return {
        "all_13_months": {
            "n_months": len(per_date),
            "mean_ic": statistics.mean(ics_all) if ics_all else None,
            "median_ic": statistics.median(ics_all) if ics_all else None,
        },
        "excluding_aug_nov_2020": {
            "n_months": len(included),
            "mean_ic": statistics.mean(ics_incl) if ics_incl else None,
            "median_ic": statistics.median(ics_incl) if ics_incl else None,
        },
        "per_month_ic_remaining_after_exclusion": {d: v["ic"] for d, v in sorted(included.items())},
        "per_month_ic_excluded_months": {d: v["ic"] for d, v in sorted(excluded_rows.items())},
    }


def main():
    with open(REPORT_PATH) as f:
        report = json.load(f)
    results = report["results"]

    targets = [
        ("historical_mean", "historical_mean"),
        ("momentum_only", "momentum_only"),
    ]
    for step in ["price_only", "price_plus_volatility", "price_plus_volume",
                 "price_plus_market_regime", "full_v1_feature_set"]:
        for model in ["ridge", "logistic", "elastic_net"]:
            targets.append((f"{step}__{model}", (step, model)))

    out = {
        "diagnostic_type": "EXPLORATORY_VALIDATION_REGIME_DIAGNOSTIC",
        "purpose": ("Determine whether negative validation IC in the linear baselines is "
                    "primarily driven by the Aug-Nov 2020 regime, or persists outside it."),
        "source_report": "PHASE4_BASELINES_REPORT.json",
        "source_generated_at": report["generated_at"],
        "excluded_months": EXCLUDED_MONTHS,
        "status": ("EXPLORATORY ONLY. Read-only re-aggregation of already-computed, "
                    "already-tuned validation-period predictions. Does not touch the locked "
                    "test set, does not change feature definitions or model configurations, "
                    "does not retune hyperparameters, and must not be used to select a model "
                    "for the final test-set evaluation or to redefine the pre-registered "
                    "success criteria."),
        "models": {},
    }

    for label, path in targets:
        entry = _entries_for(results, path)
        per_date = entry.get("per_date")
        if not per_date:
            continue
        out["models"][label] = _summarize(per_date, set(EXCLUDED_MONTHS))

    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2, default=str)

    print(f"Wrote {OUT_PATH}")
    print("EXPLORATORY ONLY -- see status field. Does not touch the locked test set.")


if __name__ == "__main__":
    main()
