"""Dataset and diagnostic report writing for leakage removal."""

from common import CLEAN_CSV, LEAKAGE_JSON, OUT_DIR
from output_io import write_json


def write_leakage_outputs(clean, *, url_only, n_rows, n_dup, conflict, removed, kept, diagnostics) -> None:
    OUT_DIR.mkdir(exist_ok=True)
    clean.to_csv(CLEAN_CSV, index=False)
    write_json(LEAKAGE_JSON, {
        "mode": "url-only" if url_only else "default",
        "n_rows": n_rows,
        "exact_duplicate_rows": n_dup,
        "conflicting_label_vectors": conflict,
        "removed": {f: {"category": c, "reason": r} for f, (c, r) in removed.items()},
        "kept": kept,
        "diagnostics": diagnostics.reset_index(names="feature").to_dict(orient="records"),
    })
    print(f"\nWrote {CLEAN_CSV.relative_to(CLEAN_CSV.parents[1])} and {LEAKAGE_JSON.name}")
