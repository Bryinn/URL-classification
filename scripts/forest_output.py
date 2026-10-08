"""Prediction tables, reports, plots, and model serialization for Random Forest."""

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, classification_report

from common import GROUP, LABEL, OUT_DIR
from output_io import write_json

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

METRIC_LABELS = {
    "accuracy": "Accuracy", "balanced_accuracy": "Balanced accuracy",
    "precision": "Precision (PPV)", "recall": "Recall (sensitivity, TPR)",
    "specificity": "Specificity (TNR)", "npv": "Negative predictive value",
    "f1": "F1 (phishing)", "f1_macro": "F1 macro", "f1_weighted": "F1 weighted", "f2": "F2",
    "false_positive_rate": "False positive rate", "false_negative_rate": "False negative rate",
    "mcc": "Matthews corr. coef.", "cohen_kappa": "Cohen's kappa", "roc_auc": "ROC-AUC",
    "pr_auc": "PR-AUC (avg. precision)", "log_loss": "Log loss", "brier_score": "Brier score",
}


def raw_predictions(index, groups, y, proba, split: str) -> pd.DataFrame:
    pred = (proba >= 0.5).astype(int)
    y = np.asarray(y)
    outcome = np.select([(y == 1) & (pred == 1), (y == 0) & (pred == 0), (y == 0) & (pred == 1)],
                        ["TP", "TN", "FP"], default="FN")
    return pd.DataFrame({
        "row_index": index, GROUP: np.asarray(groups), "split": split,
        "true_label": y, "true_class": np.where(y == 1, "phishing", "legitimate"),
        "predicted_label": pred, "predicted_class": np.where(pred == 1, "phishing", "legitimate"),
        "prob_phishing": proba.round(6), "prob_legitimate": (1 - proba).round(6),
        "correct": y == pred, "outcome": outcome,
    })


def fmt(m: dict) -> str:
    return "  ".join(f"{k}={m[k]:.4f}" for k in ("accuracy", "precision", "recall", "f1", "roc_auc"))


def format_class_report(y, proba) -> str:
    return classification_report(y, (proba >= 0.5).astype(int),
                                 target_names=["legitimate", "phishing"], digits=4)


def format_results_report(*, selected, best_params, n_train, n_test, test_metrics, class_report,
                          cv_oof_metrics, cv_mean, cv_std, references) -> str:
    def block(m: dict) -> str:
        return "\n".join(f"  {METRIC_LABELS[k]:<28}{m[k]:.4f}" for k in METRIC_LABELS)

    tn, fp, fn, tp = (test_metrics[k] for k in ("tn", "fp", "fn", "tp"))
    return "\n".join([
        "RANDOM FOREST - PHISHING WEBSITE CLASSIFICATION RESULTS",
        "=" * 60,
        "Positive class: phishing (1)   Negative class: legitimate (0)   Threshold: 0.5",
        f"Features ({len(selected)}): {', '.join(selected)}",
        f"Hyper-parameters: {best_params}",
        f"Train rows: {n_train}   Test rows: {n_test} (no duplicate rows shared with train)",
        "",
        "HELD-OUT TEST SET",
        "-" * 60,
        block(test_metrics),
        "",
        "  Confusion matrix (rows = actual, cols = predicted)",
        f"  {'':<14}{'legitimate':>12}{'phishing':>12}",
        f"  {'legitimate':<14}{tn:>12}{fp:>12}",
        f"  {'phishing':<14}{fn:>12}{tp:>12}",
        "",
        "  Per-class report",
        class_report,
        "TRAINING SET - 5-FOLD GROUP CV (out-of-fold, pooled)",
        "-" * 60,
        block(cv_oof_metrics),
        "",
        "  Per-fold mean +/- std",
        "\n".join(f"  {METRIC_LABELS[k]:<28}{cv_mean[k]:.4f} +/- {cv_std[k]:.4f}" for k in METRIC_LABELS),
        "",
        "LEAKAGE REFERENCE RUNS (same hyper-parameters, NOT the final model)",
        "-" * 60,
        *(f"  {n}\n    {fmt(m)}" for n, m in references.items()),
        "",
    ])


def write_prediction_outputs(*, train, test, selected, test_proba, oof_proba, oof_fold) -> None:
    OUT_DIR.mkdir(exist_ok=True)
    pred_test = raw_predictions(test.index, test[GROUP], test[LABEL], test_proba, "test")
    pred_test = pd.concat([pred_test, test[selected].reset_index(drop=True)], axis=1)
    pred_test.to_csv(OUT_DIR / "rf_test_predictions.csv", index=False)
    pred_oof = raw_predictions(train.index, train[GROUP], train[LABEL], oof_proba, "train_oof").assign(
        cv_fold=oof_fold)
    pred_oof = pd.concat([pred_oof, train[selected].reset_index(drop=True)], axis=1)
    pred_oof.to_csv(OUT_DIR / "rf_train_oof_predictions.csv", index=False)


def write_forest_outputs(*, model, selected, best_params, cv_roc_auc, n_train, n_test,
                         test_metrics, class_report, cv_oof_metrics, fold_rows, cv_mean, cv_std,
                         references) -> None:
    OUT_DIR.mkdir(exist_ok=True)
    joblib.dump({"model": model, "features": selected, "positive_class": "phishing"},
                OUT_DIR / "random_forest.joblib")

    importances = pd.Series(model.feature_importances_, index=selected).sort_values()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    try:
        importances.plot.barh(ax=ax)
        ax.set(title="Random Forest feature importance (impurity)", xlabel="importance")
        fig.tight_layout()
        fig.savefig(OUT_DIR / "rf_feature_importance.png", dpi=130)
    finally:
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.5, 4))
    try:
        ConfusionMatrixDisplay(np.array(test_metrics["confusion_matrix"]),
                               display_labels=["legitimate", "phishing"]).plot(ax=ax, colorbar=False)
        ax.set_title("Held-out test set")
        fig.tight_layout()
        fig.savefig(OUT_DIR / "rf_confusion_matrix.png", dpi=130)
    finally:
        plt.close(fig)

    write_json(OUT_DIR / "rf_metrics.json", {
        "features": selected,
        "best_params": best_params,
        "cv_roc_auc": cv_roc_auc,
        "test": test_metrics,
        "train_cv_out_of_fold": cv_oof_metrics,
        "train_cv_folds": fold_rows,
        "feature_importance": importances.sort_values(ascending=False).to_dict(),
        "leakage_references": references,
    })

    summary = pd.DataFrame(
        [{"evaluation": "test_holdout", **test_metrics},
         {"evaluation": "train_cv_out_of_fold", **cv_oof_metrics},
         *fold_rows, cv_mean, cv_std,
         *({"evaluation": f"reference: {n}", **m} for n, m in references.items())]
    ).drop(columns="confusion_matrix")
    summary.to_csv(OUT_DIR / "rf_metrics_summary.csv", index=False, float_format="%.6f")

    report = format_results_report(
        selected=selected, best_params=best_params, n_train=n_train, n_test=n_test,
        test_metrics=test_metrics, class_report=class_report, cv_oof_metrics=cv_oof_metrics,
        cv_mean=cv_mean, cv_std=cv_std, references=references,
    )
    (OUT_DIR / "rf_results_report.txt").write_text(report, encoding="utf-8")
    print("\nWrote random_forest.joblib, rf_metrics.json, rf_metrics_summary.csv, rf_results_report.txt,\n"
          "      rf_test_predictions.csv, rf_train_oof_predictions.csv, rf_feature_importance.png,"
          " rf_confusion_matrix.png")
