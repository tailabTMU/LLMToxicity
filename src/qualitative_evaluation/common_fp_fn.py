"""
Export rows where three toxicity detectors (Detoxify, Llama Guard, MindGuard-4B)
all share the same false positive (FP) or false negative (FN) on ToxicChat-style
parquet outputs.

Install dependencies::

    pip install -r requirements.txt

Run (example; Detoxify rule has no defaults — you must pass threshold/columns)::

    python src/qualitative_evaluation/common_fp_fn.py \
        --detoxify-parquet <detoxify_parquet_path> \
        --llama-parquet  <llama_parquet_path> \
        --mindguard-parquet  <mindguard_parquet_path> \
        --detoxify-agg single \
        --detoxify-columns detox_toxicity \
        --detoxify-threshold 0.0004540717345662415 \
        --detoxify-op ge \
        --llama1b-score-col guard_unsafe_score \
        --llama1b-threshold 0.003486792324110866 \
        --llama1b-op ge \
        --mindguard4b-score-col guard_unsafe_score \
        --mindguard4b-threshold 0.5 \
        --mindguard4b-op ge \
        --output-dir <output_dir>

Gold standard: ``label == 0`` non-toxic, ``label == 1`` toxic.

FP: label==0 and all three models predict toxic (pred==1).
FN: label==1 and all three predict non-toxic (pred==0).

Note: Check raw metrics computed by eval.py for the value of threshold
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

SHARED_CONTENT_COLS = ("dataset_name", "split", "label", "text", "text_field")
ID_COL = "sample_id"


def _eprint(*args: object) -> None:
    print(*args, file=sys.stderr)


def _parse_guard_toxic_labels(s: str) -> set[str]:
    return {part.strip().lower() for part in s.split(",") if part.strip()}


def _assert_unique_sample_ids(df: pd.DataFrame, name: str) -> None:
    if ID_COL not in df.columns:
        raise SystemExit(f"{name}: missing required column {ID_COL!r}")
    dup = df[ID_COL].duplicated()
    if dup.any():
        n = int(dup.sum())
        raise SystemExit(f"{name}: {n} duplicate {ID_COL} values")


def _assert_same_sample_id_sets(
    ids_d: set[object], ids_l: set[object], ids_m: set[object]
) -> None:
    if ids_d == ids_l == ids_m:
        return
    _eprint("sample_id sets differ across parquet files.")
    _eprint(f"  detoxify: {len(ids_d)}")
    _eprint(f"  llama:    {len(ids_l)}")
    _eprint(f"  mindguard:{len(ids_m)}")
    only_d = ids_d - ids_l - ids_m
    only_l = ids_l - ids_d - ids_m
    only_m = ids_m - ids_d - ids_l
    pair_dl = (ids_d & ids_l) - ids_m
    pair_dm = (ids_d & ids_m) - ids_l
    pair_lm = (ids_l & ids_m) - ids_d
    _eprint(f"  only in detoxify: {len(only_d)}")
    _eprint(f"  only in llama:    {len(only_l)}")
    _eprint(f"  only in mindguard:{len(only_m)}")
    _eprint(f"  detox∩llama minus mindguard: {len(pair_dl)}")
    _eprint(f"  detox∩mindguard minus llama: {len(pair_dm)}")
    _eprint(f"  llama∩mindguard minus detox: {len(pair_lm)}")
    raise SystemExit(1)


def _series_equal_with_na(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a == b) | (a.isna() & b.isna())


def _verify_shared_content(
    d: pd.DataFrame,
    l: pd.DataFrame,
    m: pd.DataFrame,
    *,
    allow_mismatch: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Index | None]:
    """Return possibly filtered (d,l,m) and index of sample_ids to keep (or None = all)."""
    d_i = d.set_index(ID_COL).sort_index()
    l_i = l.set_index(ID_COL).sort_index()
    m_i = m.set_index(ID_COL).sort_index()

    missing = [c for c in SHARED_CONTENT_COLS if c not in d_i.columns]
    if missing:
        raise SystemExit(f"detoxify parquet missing columns: {missing}")

    for name, df_i in ("llama", l_i), ("mindguard", m_i):
        miss = [c for c in SHARED_CONTENT_COLS if c not in df_i.columns]
        if miss:
            raise SystemExit(f"{name} parquet missing columns: {miss}")

    ok = pd.Series(True, index=d_i.index)
    for col in SHARED_CONTENT_COLS:
        ok &= _series_equal_with_na(d_i[col], l_i[col])
        ok &= _series_equal_with_na(d_i[col], m_i[col])

    if ok.all():
        return d, l, m, None

    bad = int((~ok).sum())
    if not allow_mismatch:
        _eprint(
            f"Shared columns {list(SHARED_CONTENT_COLS)} disagree on {bad} "
            f"{ID_COL}(s) across the three files."
        )
        _eprint("Use --allow-shared-mismatch to drop those rows and continue.")
        raise SystemExit(1)

    _eprint(
        f"Warning: dropping {bad} {ID_COL}(s) where shared columns disagree "
        f"({list(SHARED_CONTENT_COLS)})."
    )
    keep = ok[ok].index
    d_f = d[d[ID_COL].isin(keep)].copy()
    l_f = l[l[ID_COL].isin(keep)].copy()
    m_f = m[m[ID_COL].isin(keep)].copy()
    return d_f, l_f, m_f, keep


def _prefixed_side(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    x = df.set_index(ID_COL).sort_index()
    drop = [c for c in SHARED_CONTENT_COLS if c in x.columns]
    if drop:
        x = x.drop(columns=drop)
    x = x.rename(columns={c: f"{prefix}_{c}" for c in x.columns})
    return x.reset_index()


def merge_wide(d: pd.DataFrame, l: pd.DataFrame, m: pd.DataFrame) -> pd.DataFrame:
    l_p = _prefixed_side(l, "llama")
    m_p = _prefixed_side(m, "mindguard")
    d_s = d.sort_values(ID_COL).reset_index(drop=True)
    wide = d_s.merge(l_p, on=ID_COL, how="inner").merge(m_p, on=ID_COL, how="inner")
    if len(wide) != len(d_s):
        raise SystemExit(
            "Inner merge lost rows after prefixing; check duplicate keys or missing IDs."
        )
    return wide


def _detoxify_score(
    df: pd.DataFrame, columns: list[str], agg: str
) -> pd.Series:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise SystemExit(f"Detoxify columns not in merged frame: {missing}")
    subset = df[columns].apply(pd.to_numeric, errors="coerce")
    if agg == "single":
        if len(columns) != 1:
            raise SystemExit("--detoxify-agg single requires exactly one column in --detoxify-columns")
        return subset.iloc[:, 0]
    if agg == "max":
        if len(columns) < 1:
            raise SystemExit("--detoxify-agg max requires at least one column in --detoxify-columns")
        return subset.max(axis=1)
    raise SystemExit(f"Unknown --detoxify-agg: {agg!r}")


def _apply_threshold(scores: pd.Series, op: str, threshold: float) -> pd.Series:
    if op == "ge":
        pred_bool = scores >= threshold
    elif op == "gt":
        pred_bool = scores > threshold
    else:
        raise SystemExit(f"Unknown --detoxify-op: {op!r}")
    out = pred_bool.astype("Int64")
    return out.where(scores.notna(), pd.NA)


def _guard_pred(series: pd.Series, toxic_labels: set[str]) -> pd.Series:
    def norm(x: object) -> str:
        if pd.isna(x):
            return ""
        return str(x).strip().lower()

    norm_s = series.map(norm)
    return norm_s.isin(toxic_labels).astype("int64")


def _coerce_binary_label(series: pd.Series, label_col: str) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce")
    if values.isna().any():
        bad_count = int(values.isna().sum())
        raise SystemExit(
            f"Label column {label_col!r} contains {bad_count} non-numeric / missing values."
        )
    unique = sorted(pd.unique(values))
    if not set(unique).issubset({0, 1}):
        raise SystemExit(
            f"Label column {label_col!r} must be binary 0/1. Found unique values: {unique}"
        )
    return values.astype(int).to_numpy()


def _stratified_dev_eval_indices(
    df: pd.DataFrame,
    label_col: str,
    dev_fraction: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not (0.0 < dev_fraction < 1.0):
        raise SystemExit(f"dev_fraction must be in (0,1), got {dev_fraction}")

    y = _coerce_binary_label(df[label_col], label_col)
    if len(np.unique(y)) < 2:
        raise SystemExit("Cannot split a single-class dataset into DEV/EVAL stratified subsets.")

    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=1.0 - dev_fraction, random_state=random_state
    )
    dev_idx, eval_idx = next(splitter.split(df, y))
    return np.asarray(dev_idx), np.asarray(eval_idx)


def _sanity_check_thresholding(
    df: pd.DataFrame,
    *,
    detector_name: str,
    score_col_name: str,
    score_series: pd.Series,
    pred_col_name: str,
    pred_series: pd.Series,
    op: str,
    threshold: float,
    max_rows: int = 20,
) -> None:
    scores = pd.to_numeric(score_series, errors="coerce")
    if op == "ge":
        toxic_by_score = scores >= threshold
    elif op == "gt":
        toxic_by_score = scores > threshold
    else:
        raise SystemExit(f"{detector_name}: unknown op {op!r}")

    pred_not_toxic = pred_series.fillna(-999).astype("int64").ne(1)
    bad = scores.notna() & toxic_by_score & pred_not_toxic
    if not bool(bad.any()):
        return

    bad_df = df.loc[bad].copy()
    _eprint(
        f"Sanity check failed for {detector_name}: "
        f"{int(bad.sum())} rows where score {op} {threshold} but {pred_col_name} != 1."
    )
    _eprint(f"  score_col={score_col_name!r} pred_col={pred_col_name!r} op={op!r} threshold={threshold}")

    cols: list[str] = []
    for c in (ID_COL, "dataset_name", "split", "dev_eval_split", "label", "text_field", "text"):
        if c in bad_df.columns:
            cols.append(c)
    for c in (score_col_name, pred_col_name):
        if c in bad_df.columns and c not in cols:
            cols.append(c)

    preview = bad_df if max_rows <= 0 else bad_df.head(max_rows)
    if cols:
        _eprint(preview[cols].to_string(index=False, max_colwidth=120))
    else:
        _eprint(preview.to_string(index=False, max_colwidth=120))
    raise SystemExit(1)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Write CSVs of common false positives / false negatives across three parquet outputs."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--detoxify-parquet", type=Path, required=True)
    p.add_argument("--llama-parquet", type=Path, required=True)
    p.add_argument("--mindguard-parquet", type=Path, required=True)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Used when --output-fp / --output-fn are omitted.",
    )
    p.add_argument(
        "--output-fp",
        type=Path,
        default=None,
        help="Path for common false positives CSV (default: <output-dir>/common_false_positives.csv).",
    )
    p.add_argument(
        "--output-fn",
        type=Path,
        default=None,
        help="Path for common false negatives CSV (default: <output-dir>/common_false_negatives.csv).",
    )
    p.add_argument(
        "--detoxify-agg",
        choices=("single", "max"),
        required=True,
        help="How to combine Detoxify score columns before thresholding.",
    )
    p.add_argument(
        "--detoxify-columns",
        required=True,
        help="Comma-separated Detoxify numeric column names (e.g. detox_toxicity or several detox_*).",
    )
    p.add_argument(
        "--detoxify-threshold",
        type=float,
        required=True,
        help="Threshold applied to the Detoxify score (or row-wise max of columns when agg=max).",
    )
    p.add_argument(
        "--detoxify-op",
        choices=("ge", "gt"),
        required=True,
        help="If ge: toxic when score >= threshold; if gt: toxic when score > threshold.",
    )
    p.add_argument(
        "--llama1b-threshold",
        type=float,
        default=None,
        help=(
            "If set, compute Llama prediction by thresholding a numeric score column "
            "(instead of using --guard-toxic-labels on llama_guard_label)."
        ),
    )
    p.add_argument(
        "--llama1b-score-col",
        default="guard_unsafe_score",
        help=(
            "Numeric score column name in the *llama parquet* to threshold when "
            "--llama1b-threshold is set. (In the merged frame this becomes llama_<col>.)"
        ),
    )
    p.add_argument(
        "--llama1b-op",
        choices=("ge", "gt"),
        default="ge",
        help="Comparison used when thresholding the Llama score.",
    )
    p.add_argument(
        "--mindguard4b-threshold",
        type=float,
        default=None,
        help=(
            "If set, compute MindGuard prediction by thresholding a numeric score column "
            "(instead of using --guard-toxic-labels on mindguard_guard_label)."
        ),
    )
    p.add_argument(
        "--mindguard4b-score-col",
        default="guard_unsafe_score",
        help=(
            "Numeric score column name in the *mindguard parquet* to threshold when "
            "--mindguard4b-threshold is set. (In the merged frame this becomes mindguard_<col>.)"
        ),
    )
    p.add_argument(
        "--mindguard4b-op",
        choices=("ge", "gt"),
        default="ge",
        help="Comparison used when thresholding the MindGuard score.",
    )
    p.add_argument(
        "--guard-toxic-labels",
        default="unsafe",
        help="Comma-separated guard_label values (case-insensitive) that count as toxic prediction.",
    )
    p.add_argument(
        "--allow-shared-mismatch",
        action="store_true",
        help="Drop rows where shared columns differ across the three parquets instead of exiting.",
    )
    p.add_argument(
        "--dev-fraction-when-missing",
        type=float,
        default=0.20,
        help="Fraction used as DEV for the internal 20/80 stratified split.",
    )
    p.add_argument(
        "--split-random-state",
        type=int,
        default=42,
        help="Random seed for the internal stratified split.",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)

    toxic_labels = _parse_guard_toxic_labels(args.guard_toxic_labels)
    if not toxic_labels:
        raise SystemExit("--guard-toxic-labels resolved to an empty set")

    detox_cols = [c.strip() for c in args.detoxify_columns.split(",") if c.strip()]
    if not detox_cols:
        raise SystemExit("--detoxify-columns must list at least one column")

    d = pd.read_parquet(args.detoxify_parquet)
    l = pd.read_parquet(args.llama_parquet)
    m = pd.read_parquet(args.mindguard_parquet)

    _assert_unique_sample_ids(d, "detoxify")
    _assert_unique_sample_ids(l, "llama")
    _assert_unique_sample_ids(m, "mindguard")

    ids_d = set(d[ID_COL].tolist())
    ids_l = set(l[ID_COL].tolist())
    ids_m = set(m[ID_COL].tolist())
    _assert_same_sample_id_sets(ids_d, ids_l, ids_m)

    d, l, m, _ = _verify_shared_content(
        d, l, m, allow_mismatch=args.allow_shared_mismatch
    )

    wide = merge_wide(d, l, m)

    scores = _detoxify_score(wide, detox_cols, args.detoxify_agg)
    wide["detoxify_score"] = scores
    wide["pred_detoxify"] = _apply_threshold(scores, args.detoxify_op, args.detoxify_threshold)

    dev_idx, eval_idx = _stratified_dev_eval_indices(
        wide,
        label_col="label",
        dev_fraction=args.dev_fraction_when_missing,
        random_state=args.split_random_state,
    )
    wide["dev_eval_split"] = "dev"
    wide.loc[wide.index.isin(eval_idx), "dev_eval_split"] = "eval"

    if args.llama1b_threshold is None:
        if "llama_guard_label" not in wide.columns:
            raise SystemExit("Merged frame missing llama_guard_label (unexpected schema).")
        wide["pred_llama_guard"] = _guard_pred(wide["llama_guard_label"], toxic_labels)
    else:
        llama_score_col = f"llama_{args.llama1b_score_col}"
        if llama_score_col not in wide.columns:
            raise SystemExit(
                f"Merged frame missing {llama_score_col!r}. "
                "If your llama parquet uses a different score column, pass --llama1b-score-col."
            )
        llama_scores = pd.to_numeric(wide[llama_score_col], errors="coerce")
        wide["pred_llama_guard"] = _apply_threshold(llama_scores, args.llama1b_op, args.llama1b_threshold)

    if args.mindguard4b_threshold is None:
        if "mindguard_guard_label" not in wide.columns:
            raise SystemExit("Merged frame missing mindguard_guard_label (unexpected schema).")
        wide["pred_mindguard"] = _guard_pred(wide["mindguard_guard_label"], toxic_labels)
    else:
        mindguard_score_col = f"mindguard_{args.mindguard4b_score_col}"
        if mindguard_score_col not in wide.columns:
            raise SystemExit(
                f"Merged frame missing {mindguard_score_col!r}. "
                "If your mindguard parquet uses a different score column, pass --mindguard4b-score-col."
            )
        mindguard_scores = pd.to_numeric(wide[mindguard_score_col], errors="coerce")
        wide["pred_mindguard"] = _apply_threshold(
            mindguard_scores, args.mindguard4b_op, args.mindguard4b_threshold
        )

    _sanity_check_thresholding(
        wide,
        detector_name="detoxify",
        score_col_name="detoxify_score",
        score_series=wide["detoxify_score"],
        pred_col_name="pred_detoxify",
        pred_series=wide["pred_detoxify"],
        op=args.detoxify_op,
        threshold=float(args.detoxify_threshold),
    )
    if args.llama1b_threshold is not None:
        llama_score_col = f"llama_{args.llama1b_score_col}"
        _sanity_check_thresholding(
            wide,
            detector_name="llama",
            score_col_name=llama_score_col,
            score_series=wide[llama_score_col],
            pred_col_name="pred_llama_guard",
            pred_series=wide["pred_llama_guard"],
            op=args.llama1b_op,
            threshold=float(args.llama1b_threshold),
        )
    if args.mindguard4b_threshold is not None:
        mindguard_score_col = f"mindguard_{args.mindguard4b_score_col}"
        _sanity_check_thresholding(
            wide,
            detector_name="mindguard",
            score_col_name=mindguard_score_col,
            score_series=wide[mindguard_score_col],
            pred_col_name="pred_mindguard",
            pred_series=wide["pred_mindguard"],
            op=args.mindguard4b_op,
            threshold=float(args.mindguard4b_threshold),
        )

    if wide[["pred_detoxify", "pred_llama_guard", "pred_mindguard"]].isna().any().any():
        _eprint("Warning: some pred_* values are NA (likely NaN Detoxify scores); those rows are excluded from FP/FN sets.")
    pred_ok = wide[["pred_detoxify", "pred_llama_guard", "pred_mindguard"]].notna().all(axis=1)

    label = wide["label"]
    eval_mask = wide["dev_eval_split"] == "eval"
    is_fp = eval_mask & pred_ok & (label == 0) & (wide["pred_detoxify"] == 1) & (wide["pred_llama_guard"] == 1) & (wide["pred_mindguard"] == 1)
    is_fn = eval_mask & pred_ok & (label == 1) & (wide["pred_detoxify"] == 0) & (wide["pred_llama_guard"] == 0) & (wide["pred_mindguard"] == 0)

    out_fp = args.output_fp or (args.output_dir / "common_false_positives.csv")
    out_fn = args.output_fn or (args.output_dir / "common_false_negatives.csv")
    out_fp.parent.mkdir(parents=True, exist_ok=True)
    out_fn.parent.mkdir(parents=True, exist_ok=True)

    wide.loc[is_fp].to_csv(out_fp, index=False, encoding="utf-8")
    wide.loc[is_fn].to_csv(out_fn, index=False, encoding="utf-8")

    _eprint(f"Rows in merged frame: {len(wide)}")
    _eprint(
        f"Internal stratified split: dev={int((wide['dev_eval_split']=='dev').sum())} "
        f"eval={int((wide['dev_eval_split']=='eval').sum())} "
        f"(dev_fraction={args.dev_fraction_when_missing}, random_state={args.split_random_state})"
    )
    _eprint(f"Common false positives written: {int(is_fp.sum())} -> {out_fp}")
    _eprint(f"Common false negatives written: {int(is_fn.sum())} -> {out_fn}")


if __name__ == "__main__":
    main()
