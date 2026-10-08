"""Step 1 - list and remove leaky features from the UCI Phishing Websites dataset.

A feature is treated as leaky when its value is not obtainable, in a
deployment-realistic way, at the moment an unseen URL is first encountered,
or when it is derived from the same source that produced the labels.
Criteria are taken from:

* L. Ramesh, SSRN 10.2139/ssrn.7399699 - PhiUSIIL's pre-computed
  URLSimilarityIndex and fields that need a third-party service are leakage;
  inference-time features should be computable from the raw URL.
* Prasad & Chandra, Computers & Security 10.1016/j.cose.2023.103545 (PhiUSIIL) -
  similarity / reputation scores against a reference list of known sites.
* Frontiers in Computer Science 10.3389/fcomp.2026.1834407 - host-level
  leakage: near-duplicate samples on both sides of a random train/test split.

Feature provenance follows "Phishing Websites Features.docx" (Mohammad,
Thabtah & McCluskey), shipped with the dataset.

Outputs: outputs/clean_dataset.csv and outputs/leakage_report.json.
Usage:   python scripts/01_remove_leaky_features.py [--url-only]
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from common import GROUP, LABEL, SEED, SPLIT, TARGET, load_raw, row_groups
from leakage_output import write_leakage_outputs

# feature -> (category, reason)
LEAKY_FEATURES: dict[str, tuple[str, str]] = {
    "Statistical_report": (
        "label-source leakage",
        "Host is looked up in PhishTank / StopBadware top-phishing lists; PhishTank is also "
        "the source of the phishing labels, so the feature partially encodes the target.",
    ),
    "web_traffic": (
        "third-party reputation",
        "Alexa traffic rank. Legitimate samples were drawn from popular-site lists, so rank "
        "encodes how a sample was collected (same pattern as PhiUSIIL's URLSimilarityIndex).",
    ),
    "Page_Rank": (
        "third-party reputation",
        "Google PageRank lookup (service discontinued); popularity proxy tied to sampling source.",
    ),
    "Google_Index": (
        "third-party reputation",
        "Requires querying Google's index; new phishing pages are unindexed by construction.",
    ),
    "Links_pointing_to_page": (
        "third-party reputation",
        "Backlink count from an external link index; popularity proxy, unavailable at first sight.",
    ),
    "age_of_domain": (
        "WHOIS / crawl-time dependent",
        "WHOIS lookup whose value depends on when the crawl happened relative to the report "
        "date; threshold (6 months) was tuned on the labelled dataset.",
    ),
    "Domain_registeration_length": (
        "WHOIS / crawl-time dependent",
        "WHOIS expiry lookup; threshold (1 year) tuned on the labelled dataset.",
    ),
    "DNSRecord": (
        "WHOIS / crawl-time dependent",
        "Empty WHOIS/DNS record often reflects a domain already taken down after being "
        "reported, i.e. post-label information.",
    ),
    "Abnormal_URL": (
        "WHOIS / crawl-time dependent",
        "Compares the host against the WHOIS-registered identity (third-party lookup).",
    ),
    "SSLfinal_State": (
        "third-party reputation",
        "Certificate issuer checked against a hand-picked 'trusted CA' list plus certificate age "
        "threshold tuned on the dataset; a curated reputation score, not a URL/page property.",
    ),
}

# Need the page to be fetched/rendered (or the host probed). Not leakage per se, but excluded
# in --url-only mode to match the SSRN paper's dependency-free lexical inference module.
PAGE_FEATURES = [
    "Favicon", "port", "Request_URL", "URL_of_Anchor", "Links_in_tags", "SFH",
    "Submitting_to_email", "Redirect", "on_mouseover", "RightClick", "popUpWidnow", "Iframe",
]

SUSPICIOUS_AUC = 0.85


def single_feature_auc(x: pd.Series, y: pd.Series, groups: pd.Series) -> float:
    """Out-of-fold ROC-AUC using only P(phishing | feature value) learned on the train folds."""
    oof = np.zeros(len(y))
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    for tr, va in cv.split(x, y, groups):
        rates = y.iloc[tr].groupby(x.iloc[tr]).mean()
        oof[va] = x.iloc[va].map(rates).fillna(y.iloc[tr].mean()).to_numpy()
    auc = roc_auc_score(y, oof)
    return max(auc, 1 - auc)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url-only", action="store_true",
                    help="also drop page/host-content features (keep only raw-URL lexical features)")
    ap.add_argument("--test-size-folds", type=int, default=5,
                    help="hold out 1/N of the groups as the test set (default 5 -> 20%%)")
    args = ap.parse_args()

    raw = load_raw()
    X_all = raw.drop(columns=TARGET)
    y = (raw[TARGET] == -1).astype(int).rename(LABEL)
    groups = row_groups(X_all)

    n_dup = int(raw.duplicated().sum())
    conflict = int((raw.assign(g=groups).groupby("g")[TARGET].nunique() > 1).sum())
    print(f"Rows: {len(raw)}  phishing: {y.sum()}  legitimate: {(1 - y).sum()}")
    print(f"Exact duplicate rows: {n_dup} ({n_dup / len(raw):.1%})  unique vectors: {groups.nunique()}"
          f"  vectors with conflicting labels: {conflict}")

    print("\nEmpirical diagnostics (group-aware out-of-fold single-feature AUC, mutual information):")
    mi = mutual_info_classif(X_all, y, discrete_features=True, random_state=SEED)
    diag = pd.DataFrame({
        "single_feature_auc": [single_feature_auc(X_all[c], y, groups) for c in X_all.columns],
        "mutual_info": mi,
    }, index=X_all.columns).sort_values("single_feature_auc", ascending=False)
    diag["documented_leaky"] = diag.index.isin(LEAKY_FEATURES)
    diag["suspicious"] = diag.single_feature_auc >= SUSPICIOUS_AUC
    print(diag.round(4).to_string())

    to_drop = dict(LEAKY_FEATURES)
    if args.url_only:
        for f in PAGE_FEATURES:
            to_drop[f] = ("page/host content (url-only mode)",
                          "Requires fetching the page or probing the host; not computable from the URL string.")

    print(f"\nLeaky features removed ({len(to_drop)}):")
    for f, (cat, why) in to_drop.items():
        print(f"  - {f:<28} [{cat}] {why}")

    flagged = [f for f in diag.index[diag.suspicious] if f not in to_drop]
    if flagged:
        print(f"\nNote: {flagged}: strong single predictor(s) (AUC >= {SUSPICIOUS_AUC}) but have "
              "deployment-realistic provenance, so kept.")

    kept = [c for c in X_all.columns if c not in to_drop]
    print(f"\nKept features ({len(kept)}): {kept}")

    # Fixed group-aware hold-out split shared by all later steps.
    split = np.full(len(raw), "train", dtype=object)
    cv = StratifiedGroupKFold(n_splits=args.test_size_folds, shuffle=True, random_state=SEED)
    _, test_idx = next(cv.split(X_all, y, groups))
    split[test_idx] = "test"

    clean = X_all[kept].assign(**{LABEL: y, GROUP: groups, SPLIT: split})
    tr_g, te_g = set(groups[split == "train"]), set(groups[split == "test"])
    print(f"\nSplit: train={int((split == 'train').sum())} test={int((split == 'test').sum())} "
          f"groups shared between train and test: {len(tr_g & te_g)}")

    write_leakage_outputs(clean, url_only=args.url_only, n_rows=len(raw), n_dup=n_dup,
                          conflict=conflict, removed=to_drop, kept=kept, diagnostics=diag)


if __name__ == "__main__":
    main()
