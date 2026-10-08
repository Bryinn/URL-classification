"""Step 3 - train and evaluate a Random Forest phishing classifier.

* Uses the leak-free features chosen in step 2 (outputs/selected_features.json).
* Hyper-parameters are tuned with RandomizedSearchCV on the training split using
  StratifiedGroupKFold (duplicate rows never straddle folds).
* The final model is evaluated once on the held-out, group-disjoint test split.
* For reference, the same RF is also scored with the original 30 features
  (including the leaky ones) and with a naive random split, to show how much
  leakage inflates the reported numbers.

Outputs: outputs/random_forest.joblib, outputs/rf_metrics.json, outputs/rf_metrics_summary.csv,
         outputs/rf_results_report.txt, outputs/rf_test_predictions.csv,
         outputs/rf_train_oof_predictions.csv, outputs/rf_feature_importance.png,
         outputs/rf_confusion_matrix.png
Usage:   python scripts/03_train_random_forest.py [--n-iter 40]
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.stats import randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, brier_score_loss,
                             cohen_kappa_score, confusion_matrix, f1_score, fbeta_score, log_loss,
                             matthews_corrcoef, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold, train_test_split

from common import (GROUP, LABEL, SEED, SELECTION_JSON, SPLIT, TARGET, load_clean,
                    load_raw, read_json)
from forest_output import fmt, format_class_report, write_forest_outputs, write_prediction_outputs

PARAM_DIST = {
    "n_estimators": randint(200, 1000),
    "max_depth": [None, 8, 12, 16, 24],
    "min_samples_leaf": randint(1, 10),
    "min_samples_split": randint(2, 20),
    "max_features": ["sqrt", "log2", 0.5, 0.75, None],
    "class_weight": [None, "balanced"],
    "criterion": ["gini", "entropy"],
}


def compute_metrics(y, proba, threshold: float = 0.5) -> dict:
    """Binary metrics with phishing (1) as the positive class."""
    y = np.asarray(y)
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    safe = lambda a, b: float(a / b) if b else float("nan")  # noqa: E731
    return {
        "n_samples": int(len(y)),
        "n_phishing": int(y.sum()),
        "n_legitimate": int((1 - y).sum()),
        "threshold": threshold,
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "specificity": safe(tn, tn + fp),
        "npv": safe(tn, tn + fn),
        "f1": f1_score(y, pred, zero_division=0),
        "f1_macro": f1_score(y, pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y, pred, average="weighted", zero_division=0),
        "f2": fbeta_score(y, pred, beta=2, zero_division=0),
        "false_positive_rate": safe(fp, fp + tn),
        "false_negative_rate": safe(fn, fn + tp),
        "mcc": matthews_corrcoef(y, pred),
        "cohen_kappa": cohen_kappa_score(y, pred),
        "roc_auc": roc_auc_score(y, proba),
        "pr_auc": average_precision_score(y, proba),
        "log_loss": log_loss(y, np.clip(proba, 1e-15, 1 - 1e-15), labels=[0, 1]),
        "brier_score": brier_score_loss(y, proba),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def evaluate(model, X, y) -> dict:
    return compute_metrics(y, model.predict_proba(X)[:, 1])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-iter", type=int, default=40, help="random-search iterations (default 40)")
    args = ap.parse_args()

    df = load_clean()
    selected = read_json(SELECTION_JSON)["selected_features"]
    train, test = df[df[SPLIT] == "train"], df[df[SPLIT] == "test"]
    X_tr, y_tr, g_tr = train[selected], train[LABEL], train[GROUP]
    X_te, y_te = test[selected], test[LABEL]
    print(f"Features ({len(selected)}): {selected}")
    print(f"Train: {len(train)} rows  Test: {len(test)} rows (group-disjoint)")

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    search = RandomizedSearchCV(
        RandomForestClassifier(random_state=SEED, n_jobs=-1), PARAM_DIST, n_iter=args.n_iter,
        scoring="roc_auc", cv=cv, random_state=SEED, n_jobs=1, refit=True,
    )
    search.fit(X_tr, y_tr, groups=g_tr)
    rf = search.best_estimator_
    print(f"\nBest params: {search.best_params_}")
    print(f"Group-CV ROC-AUC (train): {search.best_score_:.4f}")

    test_proba = rf.predict_proba(X_te)[:, 1]
    test_metrics = compute_metrics(y_te, test_proba)
    print(f"\nHeld-out test: {fmt(test_metrics)}")
    class_report = format_class_report(y_te, test_proba)
    print(class_report)

    # Out-of-fold predictions on the training split with the tuned hyper-parameters.
    params = {**search.best_params_, "random_state": SEED, "n_jobs": -1}
    oof_proba, oof_fold, fold_rows = np.zeros(len(train)), np.zeros(len(train), dtype=int), []
    for k, (tr_i, va_i) in enumerate(cv.split(X_tr, y_tr, g_tr), start=1):
        m = RandomForestClassifier(**params).fit(X_tr.iloc[tr_i], y_tr.iloc[tr_i])
        oof_proba[va_i] = m.predict_proba(X_tr.iloc[va_i])[:, 1]
        oof_fold[va_i] = k
        fold_rows.append({"evaluation": f"cv_fold_{k}", **compute_metrics(y_tr.iloc[va_i], oof_proba[va_i])})
    cv_oof_metrics = compute_metrics(y_tr, oof_proba)
    fold_df = pd.DataFrame(fold_rows)
    num_cols = [c for c in fold_df.columns if c not in ("evaluation", "confusion_matrix")]
    cv_mean = {"evaluation": "cv_mean", **fold_df[num_cols].mean().to_dict()}
    cv_std = {"evaluation": "cv_std", **fold_df[num_cols].std(ddof=1).to_dict()}
    print(f"Train out-of-fold (5-fold group CV): {fmt(cv_oof_metrics)}")

    # Raw per-sample outputs.
    write_prediction_outputs(train=train, test=test, selected=selected, test_proba=test_proba,
                             oof_proba=oof_proba, oof_fold=oof_fold)

    # Reference runs quantifying leakage, using the same tuned hyper-parameters.
    raw = load_raw()
    X_raw, y_raw = raw.drop(columns=TARGET), (raw[TARGET] == -1).astype(int)
    is_tr = (df[SPLIT] == "train").to_numpy()

    def ref(cols, tr_mask=None):
        if tr_mask is None:
            idx_tr, idx_te = train_test_split(np.arange(len(raw)), test_size=0.2,
                                              stratify=y_raw, random_state=SEED)
        else:
            idx_tr, idx_te = np.where(tr_mask)[0], np.where(~tr_mask)[0]
        m = RandomForestClassifier(**params).fit(X_raw.iloc[idx_tr][cols], y_raw.iloc[idx_tr])
        return evaluate(m, X_raw.iloc[idx_te][cols], y_raw.iloc[idx_te])

    all30 = list(X_raw.columns)
    references = {
        "all_30_features_random_split (leaky features + duplicate leakage)": ref(all30),
        "all_30_features_group_split (leaky features)": ref(all30, is_tr),
        "selected_features_random_split (duplicate leakage)": ref(selected),
    }
    print("Leakage reference runs (same RF hyper-parameters):")
    for name, m in references.items():
        print(f"  {name}\n      {fmt(m)}")
    print(f"  selected_features_group_split (this model)\n      {fmt(test_metrics)}")

    write_forest_outputs(
        model=rf, selected=selected, best_params=search.best_params_, cv_roc_auc=search.best_score_,
        n_train=len(train), n_test=len(test), test_metrics=test_metrics, class_report=class_report,
        cv_oof_metrics=cv_oof_metrics, fold_rows=fold_rows, cv_mean=cv_mean, cv_std=cv_std,
        references=references,
    )


if __name__ == "__main__":
    main()
