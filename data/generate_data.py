"""
Synthetic maintenance-risk dataset generator.

Simulates task-level records like what would come out of TMS / SMMS / TDMS,
plus a *ground truth* outcome (did the asset actually fail / cause disruption
before it was attended to) generated from a latent risk function + noise.

Why this design:
- We don't hand-label "priority" (that would make the model circular).
- Instead we simulate a latent risk, then draw a stochastic binary outcome
  from it (like a real asset would: high risk doesn't guarantee failure,
  it just makes it more likely). XGBoost is trained to predict THAT outcome
  from the observable features -> that predicted probability becomes the
  learned risk/priority score.
- Two independent batches are produced (TRAIN, OOS_TEST) with different
  random seeds and a mild distribution shift, so generalization can be
  checked honestly instead of testing on data the model has effectively seen.
"""

import numpy as np
import pandas as pd

DEPARTMENTS = ["ENG", "SNT", "TRD"]
ASSET_TYPES = {
    "ENG": ["Rail", "Ballast", "Bridge", "Fastening"],
    "SNT": ["Interlocking", "Signal LED", "Axle Counter", "Point Machine", "Cable"],
    "TRD": ["OHE", "Sub-station", "Feeder", "Insulator"],
}
DEFECT_TYPES = {
    "Rail": "Rail crack", "Ballast": "Ballast deficiency", "Bridge": "Bearing wear",
    "Fastening": "Fastening renewal", "Interlocking": "Interlocking fault",
    "Signal LED": "LED failure", "Axle Counter": "Axle counter drift",
    "Point Machine": "Point machine overdue", "Cable": "Cable insulation fault",
    "OHE": "OHE wear", "Sub-station": "Sub-station overdue PM",
    "Feeder": "Feeder overdue PM", "Insulator": "Insulator crack",
}


def _gen_batch(n, seed, era_shift=0.0):
    rng = np.random.default_rng(seed)

    dept = rng.choice(DEPARTMENTS, size=n, p=[0.45, 0.30, 0.25])
    asset_type = np.array([rng.choice(ASSET_TYPES[d]) for d in dept])
    defect_type = np.array([DEFECT_TYPES[a] for a in asset_type])

    severity = rng.integers(1, 6, size=n)                       # 1-5
    days_overdue = np.clip(rng.exponential(scale=8, size=n) + era_shift * 3, 0, 120).round().astype(int)
    asset_age_years = np.clip(rng.normal(12, 6, size=n), 0, 45).round(1)
    past_failures_12m = rng.poisson(lam=0.35 + severity * 0.08, size=n)
    traffic_density = np.clip(rng.normal(60, 25, size=n), 5, 160).round().astype(int)  # trains/day on section
    redundancy_single_line = rng.choice([0, 1], size=n, p=[0.55, 0.45])                 # 1 = single line, more critical
    safety_critical = np.isin(asset_type, ["Interlocking", "Rail", "OHE", "Point Machine", "Axle Counter"]).astype(int)
    weather_exposure = rng.choice(["Low", "Medium", "High"], size=n, p=[0.5, 0.35, 0.15])
    weather_score = pd.Series(weather_exposure).map({"Low": 0, "Medium": 1, "High": 2}).to_numpy()
    days_since_last_maint = np.clip(rng.normal(180, 90, size=n), 5, 900).round().astype(int)
    block_duration_min = np.clip(rng.normal(55, 20, size=n), 20, 150).round().astype(int)

    # ---- latent risk (ground-truth generating process, hidden from the model at inference) ----
    z = (
        0.55 * severity
        + 0.05 * days_overdue
        + 0.35 * past_failures_12m
        + 0.02 * asset_age_years
        + 0.015 * traffic_density
        + 1.1 * redundancy_single_line
        + 1.3 * safety_critical
        + 0.4 * weather_score
        + 0.004 * days_since_last_maint
        - 8.9                                    # intercept so base rate is realistic (~10-20%)
        + era_shift                              # small distribution shift for OOS batch
    )
    z += rng.normal(0, 0.35, size=n)             # irreducible noise -> label is NOT a deterministic function of features
    prob_incident = 1 / (1 + np.exp(-z / 1.6))
    incident_occurred = rng.binomial(1, prob_incident)

    df = pd.DataFrame({
        "task_id": [f"T{seed}{i:05d}" for i in range(n)],
        "department": dept,
        "asset_type": asset_type,
        "defect_type": defect_type,
        "severity": severity,
        "days_overdue": days_overdue,
        "asset_age_years": asset_age_years,
        "past_failures_12m": past_failures_12m,
        "traffic_density_trains_per_day": traffic_density,
        "single_line_section": redundancy_single_line,
        "safety_critical_asset": safety_critical,
        "weather_exposure": weather_exposure,
        "days_since_last_maintenance": days_since_last_maint,
        "block_duration_min": block_duration_min,
        "incident_occurred": incident_occurred,     # <-- training label
        "true_incident_probability": prob_incident.round(4),  # kept only for our own sanity-check, not a model input
    })
    return df


def generate(train_n=4000, oos_n=1200, seed=42):
    train_df = _gen_batch(train_n, seed=seed, era_shift=0.0)
    # OOS batch: different seed AND a mild distribution shift (more overdue backlog, slightly higher risk era)
    # to genuinely test generalization rather than re-sampling the same distribution.
    oos_df = _gen_batch(oos_n, seed=seed + 999, era_shift=0.6)
    return train_df, oos_df


if __name__ == "__main__":
    train_df, oos_df = generate()
    train_df.to_csv("/home/claude/train_data.csv", index=False)
    oos_df.to_csv("/home/claude/oos_test_data.csv", index=False)
    print("TRAIN:", train_df.shape, "incident rate:", train_df.incident_occurred.mean().round(3))
    print("OOS  :", oos_df.shape, "incident rate:", oos_df.incident_occurred.mean().round(3))
    print(train_df.head(3).to_string())
