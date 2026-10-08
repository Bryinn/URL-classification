"""Step 2 - GPU feature selection on the leak-free training split.

Recursive feature elimination driven by SHAP values:
  * model: XGBoost random-forest mode (XGBRFClassifier) on CUDA, so the wrapper
    uses the same model family as the final Random Forest;
  * at every step, 5-fold StratifiedGroupKFold (duplicate rows never straddle folds)
    gives the CV ROC-AUC and mean |SHAP| (computed on GPU via pred_contribs) on
    the validation folds; the least important feature is dropped;
  * the selected subset is the smallest one whose CV AUC is within one standard
    error of the best (1-SE rule).
The test split is never touched here.

Outputs: outputs/selected_features.json, outputs/feature_selection_curve.png
Usage:   python scripts/02_select_features.py [--device cuda|cpu]
"""

from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from common import GROUP, LABEL, SEED, SPLIT, feature_cols, load_clean
from selection_output import write_selection_outputs

warnings.filterwarnings("ignore", message=".*Falling back to prediction using DMatrix.*")
N_FOLDS = 5


def make_model(device: str) -> xgb.XGBRFClassifier:
    return xgb.XGBRFClassifier(
        n_estimators=300, max_depth=10, subsample=0.8, colsample_bynode=0.6,
        tree_method="hist", device=device, random_state=SEED, n_jobs=-1,
    )


def cv_step(X: pd.DataFrame, y: pd.Series, groups: pd.Series, device: str):
    cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    aucs, shap_abs = [], np.zeros(X.shape[1])
    for tr, va in cv.split(X, y, groups):
        model = make_model(device).fit(X.iloc[tr], y.iloc[tr])
        aucs.append(roc_auc_score(y.iloc[va], model.predict_proba(X.iloc[va])[:, 1]))
        contribs = model.get_booster().predict(xgb.DMatrix(X.iloc[va]), pred_contribs=True)
        shap_abs += np.abs(contribs[:, :-1]).mean(axis=0)  # last column is the bias term
    return np.array(aucs), pd.Series(shap_abs / N_FOLDS, index=X.columns)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda", help="xgboost device (default: cuda)")
    args = ap.parse_args()

    df = load_clean()
    train = df[df[SPLIT] == "train"].reset_index(drop=True)
    feats = feature_cols(df)
    X, y, groups = train[feats], train[LABEL], train[GROUP]
    print(f"Training rows: {len(train)}  candidate features: {len(feats)}  device: {args.device}")

    mi = pd.Series(mutual_info_classif(X, y, discrete_features=True, random_state=SEED), index=feats)

    remaining, history, elimination_order, initial_shap = list(feats), [], [], None
    while remaining:
        aucs, shap = cv_step(X[remaining], y, groups, args.device)
        if initial_shap is None:
            initial_shap = shap
        history.append({"n_features": len(remaining), "auc_mean": aucs.mean(),
                        "auc_se": aucs.std(ddof=1) / np.sqrt(N_FOLDS), "features": list(remaining)})
        weakest = shap.idxmin()
        print(f"  k={len(remaining):2d}  CV AUC={aucs.mean():.4f} +/- {history[-1]['auc_se']:.4f}"
              f"  drop next: {weakest} (|SHAP|={shap.min():.4f})", flush=True)
        elimination_order.append(weakest)
        remaining.remove(weakest)

    curve = pd.DataFrame(history)
    best = curve.loc[curve.auc_mean.idxmax()]
    threshold = best.auc_mean - best.auc_se
    chosen = curve[curve.auc_mean >= threshold].sort_values("n_features").iloc[0]
    selected = chosen.features

    # Elimination order reversed = importance rank (last survivor is most important).
    ranking = pd.DataFrame({
        "rfe_rank": {f: i + 1 for i, f in enumerate(reversed(elimination_order))},
        "initial_mean_abs_shap": initial_shap,
        "mutual_info": mi,
    }).sort_values("rfe_rank")
    ranking["selected"] = ranking.index.isin(selected)
    print("\nFeature ranking:\n" + ranking.round(4).to_string())
    print(f"\nBest CV AUC {best.auc_mean:.4f} at k={best.n_features}; 1-SE threshold {threshold:.4f}")
    print(f"Selected {len(selected)} features (CV AUC {chosen.auc_mean:.4f}): {selected}")
    dropped = [f for f in feats if f not in selected]
    print(f"Dropped by selection ({len(dropped)}): {dropped}")

    write_selection_outputs(device=args.device, selected=selected, dropped=dropped,
                            best=best, chosen=chosen, ranking=ranking, curve=curve, threshold=threshold)


if __name__ == "__main__":
    main()
