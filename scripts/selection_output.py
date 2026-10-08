"""Plotting and report writing for feature selection."""

import matplotlib

from common import OUT_DIR, SELECTION_JSON
from output_io import write_json

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def write_selection_outputs(*, device, selected, dropped, best, chosen, ranking, curve, threshold) -> None:
    OUT_DIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    try:
        ax.errorbar(curve.n_features, curve.auc_mean, yerr=curve.auc_se, marker="o", ms=4, capsize=3)
        ax.axhline(threshold, ls="--", c="grey", label="best - 1 SE")
        ax.axvline(chosen.n_features, ls=":", c="red", label=f"selected k={chosen.n_features}")
        ax.set(xlabel="number of features", ylabel="group-CV ROC-AUC",
               title="SHAP-RFE with GPU random forest (XGBRF)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUT_DIR / "feature_selection_curve.png", dpi=130)
    finally:
        plt.close(fig)

    write_json(SELECTION_JSON, {
        "method": "SHAP-based RFE, XGBRFClassifier on " + device + ", 5-fold StratifiedGroupKFold, 1-SE rule",
        "selected_features": selected,
        "dropped_features": dropped,
        "best": {"n_features": int(best.n_features), "cv_auc": best.auc_mean},
        "chosen": {"n_features": int(chosen.n_features), "cv_auc": chosen.auc_mean},
        "ranking": ranking.reset_index(names="feature").to_dict(orient="records"),
        "curve": curve.drop(columns="features").to_dict(orient="records"),
    })
    print(f"\nWrote {SELECTION_JSON.name} and feature_selection_curve.png")
