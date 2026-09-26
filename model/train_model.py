"""
Train an XGBoost risk-prioritization model on the synthetic maintenance data,
validate it with stratified k-fold cross-validation (for development confidence),
then do a final check against a held-out, independently-generated OOS batch
(for generalization confidence).
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                              precision_score, recall_score, brier_score_loss,
                              roc_curve, confusion_matrix)
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

FEATURES = [
    "severity", "days_overdue", "asset_age_years", "past_failures_12m",
    "traffic_density_trains_per_day", "single_line_section", "safety_critical_asset",
    "weather_score", "days_since_last_maintenance", "block_duration_min",
    "department_enc", "asset_type_enc",
]
LABEL = "incident_occurred"


def prep(df, dept_enc=None, asset_enc=None, fit=False):
    df = df.copy()
    df["weather_score"] = df["weather_exposure"].map({"Low": 0, "Medium": 1, "High": 2})
    if fit:
        dept_enc, asset_enc = LabelEncoder(), LabelEncoder()
        df["department_enc"] = dept_enc.fit_transform(df["department"])
        df["asset_type_enc"] = asset_enc.fit_transform(df["asset_type"])
    else:
        df["department_enc"] = dept_enc.transform(df["department"])
        # unseen asset types in OOS -> map to -1 safely
        df["asset_type_enc"] = df["asset_type"].map(
            {cls: i for i, cls in enumerate(asset_enc.classes_)}
        ).fillna(-1).astype(int)
    return df, dept_enc, asset_enc


def main():
    train_df = pd.read_csv("/home/claude/train_data.csv")
    oos_df = pd.read_csv("/home/claude/oos_test_data.csv")

    train_df, dept_enc, asset_enc = prep(train_df, fit=True)
    oos_df, _, _ = prep(oos_df, dept_enc, asset_enc, fit=False)

    X, y = train_df[FEATURES], train_df[LABEL]
    X_oos, y_oos = oos_df[FEATURES], oos_df[LABEL]

    params = dict(
        n_estimators=250, max_depth=4, learning_rate=0.05,
        subsample=0.85, colsample_bytree=0.85,
        eval_metric="auc", random_state=42,
    )

    # ---------------- Stratified 5-fold CV on the training set ----------------
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_metrics = []
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        model = xgb.XGBClassifier(**params)
        model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        p = model.predict_proba(X.iloc[val_idx])[:, 1]
        pred = (p >= 0.5).astype(int)
        yv = y.iloc[val_idx]
        fold_metrics.append({
            "fold": fold,
            "auc": roc_auc_score(yv, p),
            "pr_auc": average_precision_score(yv, p),
            "f1": f1_score(yv, pred),
            "precision": precision_score(yv, pred),
            "recall": recall_score(yv, pred),
            "brier": brier_score_loss(yv, p),
        })
    cv_df = pd.DataFrame(fold_metrics)

    # ---------------- Final model on ALL training data ----------------
    final_model = xgb.XGBClassifier(**params)
    final_model.fit(X, y)

    # ---------------- Independent OOS test (never touched during training/CV) ----------------
    p_oos = final_model.predict_proba(X_oos)[:, 1]
    pred_oos = (p_oos >= 0.5).astype(int)
    oos_metrics = {
        "auc": roc_auc_score(y_oos, p_oos),
        "pr_auc": average_precision_score(y_oos, p_oos),
        "f1": f1_score(y_oos, pred_oos),
        "precision": precision_score(y_oos, pred_oos),
        "recall": recall_score(y_oos, pred_oos),
        "brier": brier_score_loss(y_oos, p_oos),
    }
    cm = confusion_matrix(y_oos, pred_oos)

    # ---------------- Save everything ----------------
    final_model.save_model("/home/claude/xgb_risk_model.json")

    report = {
        "cv_per_fold": cv_df.round(4).to_dict(orient="records"),
        "cv_mean": cv_df.drop(columns="fold").mean().round(4).to_dict(),
        "cv_std": cv_df.drop(columns="fold").std().round(4).to_dict(),
        "oos_test_metrics": {k: round(v, 4) for k, v in oos_metrics.items()},
        "oos_confusion_matrix": cm.tolist(),
        "feature_importance": dict(sorted(
            zip(FEATURES, final_model.feature_importances_.round(4).tolist()),
            key=lambda x: -x[1]
        )),
        "n_train": len(train_df), "n_oos_test": len(oos_df),
        "train_incident_rate": round(float(y.mean()), 4),
        "oos_incident_rate": round(float(y_oos.mean()), 4),
    }
    with open("/home/claude/model_report.json", "w") as f:
        json.dump(report, f, indent=2)

    # score every train task with the priority model -> save as scored dataset
    train_df["ai_priority_score"] = final_model.predict_proba(X)[:, 1].round(4)
    train_df.sort_values("ai_priority_score", ascending=False)\
        .drop(columns=["department_enc", "asset_type_enc", "weather_score"])\
        .to_csv("/home/claude/scored_tasks.csv", index=False)

    # ---------------- Plots ----------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    imp = report["feature_importance"]
    axes[0].barh(list(imp.keys())[::-1], list(imp.values())[::-1], color="#2f6fed")
    axes[0].set_title("Feature importance (XGBoost gain)")

    fpr, tpr, _ = roc_curve(y_oos, p_oos)
    axes[1].plot(fpr, tpr, color="#e14545", label=f"OOS AUC={oos_metrics['auc']:.3f}")
    axes[1].plot([0, 1], [0, 1], "--", color="gray")
    axes[1].set_title("ROC curve — independent OOS set")
    axes[1].set_xlabel("False positive rate"); axes[1].set_ylabel("True positive rate")
    axes[1].legend()

    axes[2].bar(["CV fold " + str(i) for i in cv_df.fold] + ["OOS"],
                list(cv_df.auc) + [oos_metrics["auc"]],
                color=["#a9cdf2"] * len(cv_df) + ["#e14545"])
    axes[2].set_ylim(0.5, 1.0)
    axes[2].set_title("AUC: 5-fold CV vs held-out OOS")
    axes[2].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig("/home/claude/model_validation.png", dpi=140)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
