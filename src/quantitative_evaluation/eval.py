'''
python src/quantitative_evaluation/eval.py \
  --test <test_file_path>  \
  --dev  <dev_file_path> \
  --out-dir detoxify_scores_results \
  --target-recall 0.95 \
  --n-balance-repeats 30
'''

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit


# Backward-compatible Detoxify heads.
CANONICAL_DETOX_HEADS = [
    "detox_toxicity",
    "detox_severe_toxicity",
    "detox_obscene",
    "detox_threat",
    "detox_insult",
    "detox_identity_attack",
]

# Generic score columns. Higher score must mean "more unsafe / more toxic".
SCORE_CANDIDATES = [
    "unsafe_score",
    "guard_unsafe_score",
    "toxicity_score",
    "detox_toxicity",
]

LABEL_CANDIDATES = ["label", "toxicity", "toxicity_label"]
TEXT_CANDIDATES = ["text", "user_input", "prompt", "response"]
ID_CANDIDATES = ["sample_id", "conv_id", "id"]

# If a Detoxify-format file is detected and --score-col is not provided,
# the script evaluates these three scalarizations for backward compatibility.
DETOX_SCORING_NAMES = ["toxicity_only", "mean_all_dims", "max_all_dims"]


@dataclass
class EvalConfig:
    dev_fraction_when_missing: float = 0.20
    split_random_state: int = 42
    target_recall: float = 0.95
    n_balance_repeats: int = 30
    balance_random_seed: int = 1234
    balance_label_strategy: str = "downsample_majority"
    file_ext: str = ".parquet"
    save_outputs: bool = True
    score_col: Optional[str] = None


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file type: {path}")


def find_existing_column(columns: Iterable[str], candidates: Sequence[str]) -> Optional[str]:
    colset = set(columns)
    for c in candidates:
        if c in colset:
            return c
    return None


def detect_label_column(df: pd.DataFrame, user_label_col: Optional[str] = None) -> str:
    if user_label_col is not None:
        if user_label_col not in df.columns:
            raise KeyError(f"Provided label_col '{user_label_col}' not found.")
        return user_label_col
    label_col = find_existing_column(df.columns, LABEL_CANDIDATES)
    if label_col is None:
        raise KeyError(
            "Could not detect label column. Expected one of: "
            f"{LABEL_CANDIDATES}. Available columns: {list(df.columns)}"
        )
    return label_col


def detect_text_column(df: pd.DataFrame) -> Optional[str]:
    return find_existing_column(df.columns, TEXT_CANDIDATES)


def detect_id_column(df: pd.DataFrame) -> Optional[str]:
    return find_existing_column(df.columns, ID_CANDIDATES)


def detect_detox_score_columns(df: pd.DataFrame, require_toxicity: bool = True) -> List[str]:
    existing = [c for c in CANONICAL_DETOX_HEADS if c in df.columns]
    if require_toxicity and "detox_toxicity" not in existing:
        raise KeyError(
            "Missing required score column 'detox_toxicity'. "
            f"Available columns: {list(df.columns)}"
        )
    if len(existing) == 0:
        fallback = sorted([c for c in df.columns if c.startswith("detox_")])
        if not fallback:
            raise KeyError(
                "Could not detect Detoxify score columns. Expected columns starting with 'detox_'."
            )
        return fallback
    return existing


def detect_score_specs(df: pd.DataFrame, user_score_col: Optional[str] = None) -> List[Dict[str, object]]:
    """
    Return one or more score specs. Each spec tells the evaluator how to compute a
    scalar unsafe score from the dataframe.

    A spec has:
      - scoring: human-readable scoring name
      - mode: "column" or one of the Detoxify aggregate modes
      - score_col: column name for mode="column"
      - score_cols: Detoxify columns for aggregate modes

    Higher score must always mean more unsafe / more toxic.
    """
    if user_score_col is not None:
        if user_score_col not in df.columns:
            raise KeyError(
                f"Provided score_col '{user_score_col}' not found. "
                f"Available columns: {list(df.columns)}"
            )
        return [{
            "scoring": user_score_col,
            "mode": "column",
            "score_col": user_score_col,
            "score_cols": [user_score_col],
        }]

    # Generic detector / guardrail path.
    generic_col = find_existing_column(df.columns, SCORE_CANDIDATES)
    if generic_col is not None and generic_col != "detox_toxicity":
        score_name = str(df["score_name"].iloc[0]) if "score_name" in df.columns and len(df) else generic_col
        return [{
            "scoring": score_name,
            "mode": "column",
            "score_col": generic_col,
            "score_cols": [generic_col],
        }]

    # Backward-compatible Detoxify path.
    if "detox_toxicity" in df.columns:
        detox_cols = detect_detox_score_columns(df)
        return [
            {
                "scoring": "toxicity_only",
                "mode": "detox_toxicity_only",
                "score_col": "detox_toxicity",
                "score_cols": detox_cols,
            },
            {
                "scoring": "mean_all_dims",
                "mode": "detox_mean_all_dims",
                "score_col": None,
                "score_cols": detox_cols,
            },
            {
                "scoring": "max_all_dims",
                "mode": "detox_max_all_dims",
                "score_col": None,
                "score_cols": detox_cols,
            },
        ]

    raise KeyError(
        "Could not detect a score column. Provide --score-col, or include one of "
        f"{SCORE_CANDIDATES}, or provide Detoxify columns: {CANONICAL_DETOX_HEADS}."
    )


def coerce_binary_label(series: pd.Series, label_col: str) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce")
    if values.isna().any():
        bad_count = int(values.isna().sum())
        raise ValueError(
            f"Label column '{label_col}' contains {bad_count} non-numeric / missing values. "
            "Please clean labels before evaluation."
        )
    unique = sorted(pd.unique(values))
    if not set(unique).issubset({0, 1}):
        raise ValueError(
            f"Label column '{label_col}' must be binary 0/1. Found unique values: {unique}"
        )
    return values.astype(int).to_numpy()


def validate_required_columns(
    df: pd.DataFrame,
    label_col: Optional[str] = None,
    score_col: Optional[str] = None,
) -> Dict[str, object]:
    label_col = detect_label_column(df, label_col)
    score_specs = detect_score_specs(df, score_col)
    info: Dict[str, object] = {
        "label_col": label_col,
        "text_col": detect_text_column(df),
        "id_col": detect_id_column(df),
        "score_specs": score_specs,
    }
    _ = coerce_binary_label(df[label_col], label_col)

    seen_score_cols = sorted({
        c
        for spec in score_specs
        for c in (spec.get("score_cols") or [])
        if c is not None
    })
    for c in seen_score_cols:
        vals = pd.to_numeric(df[c], errors="coerce")
        if vals.isna().any():
            raise ValueError(f"Score column '{c}' contains missing / non-numeric values.")
    return info


def compute_scalar_score(df: pd.DataFrame, score_spec: Dict[str, object]) -> np.ndarray:
    mode = str(score_spec["mode"])
    if mode == "column":
        score_col = str(score_spec["score_col"])
        return pd.to_numeric(df[score_col], errors="raise").to_numpy(dtype=float)

    score_cols = list(score_spec["score_cols"])  # type: ignore[arg-type]
    if mode == "detox_toxicity_only":
        return pd.to_numeric(df["detox_toxicity"], errors="raise").to_numpy(dtype=float)

    mat = df[score_cols].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    if mode == "detox_mean_all_dims":
        return mat.mean(axis=1)
    if mode == "detox_max_all_dims":
        return mat.max(axis=1)

    raise ValueError(f"Unknown score spec mode: {mode}")


def safe_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def safe_auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def compute_threshold_free_metrics(y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    return {
        "auroc": safe_auroc(y_true, y_score),
        "auprc": safe_auprc(y_true, y_score),
        "pos_rate": float(np.mean(y_true)) if len(y_true) else float("nan"),
        "n": int(len(y_true)),
    }


def select_threshold_for_target_recall(
    y_true_dev: np.ndarray,
    y_score_dev: np.ndarray,
    target_recall: float = 0.95,
) -> float:
    if not (0.0 <= target_recall <= 1.0):
        raise ValueError(f"target_recall must be in [0,1], got {target_recall}")
    pos_scores = y_score_dev[y_true_dev == 1]
    if len(pos_scores) == 0:
        raise ValueError("No positive samples in DEV set. Cannot select threshold.")
    threshold = float(np.quantile(pos_scores, 1.0 - target_recall))
    # Do not clip: some models may output logits/log-likelihood margins rather than probabilities.
    return threshold


def compute_thresholded_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> Dict[str, float]:
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else float("nan")
    f2 = 5 * precision * recall / (4 * precision + recall) if (4 * precision + recall) > 0 else float("nan")
    fpr = fp / (fp + tn) if (fp + tn) > 0 else float("nan")
    fnr = fn / (fn + tp) if (fn + tp) > 0 else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")

    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "f2": float(f2),
        "fpr": float(fpr),
        "fnr": float(fnr),
        "specificity": float(specificity),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def stratified_dev_eval_split(
    df: pd.DataFrame,
    label_col: str,
    dev_fraction: float,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not (0.0 < dev_fraction < 1.0):
        raise ValueError(f"dev_fraction must be in (0,1), got {dev_fraction}")

    y = coerce_binary_label(df[label_col], label_col)
    if len(np.unique(y)) < 2:
        raise ValueError("Cannot split a single-class dataset into DEV/EVAL stratified subsets.")

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=1.0 - dev_fraction, random_state=random_state)
    dev_idx, eval_idx = next(splitter.split(df, y))
    df_dev = df.iloc[dev_idx].reset_index(drop=True)
    df_eval = df.iloc[eval_idx].reset_index(drop=True)
    return df_dev, df_eval


def downsample_to_balanced(
    df: pd.DataFrame,
    label_col: str,
    random_state: int,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    y = coerce_binary_label(df[label_col], label_col)
    df_pos = df.loc[y == 1].copy()
    df_neg = df.loc[y == 0].copy()

    n_pos = len(df_pos)
    n_neg = len(df_neg)
    if n_pos == 0 or n_neg == 0:
        raise ValueError("Balanced evaluation requires both positive and negative samples.")

    minority_n = min(n_pos, n_neg)
    df_pos_s = df_pos.sample(n=minority_n, replace=False, random_state=random_state)
    df_neg_s = df_neg.sample(n=minority_n, replace=False, random_state=random_state)
    out = (
        pd.concat([df_pos_s, df_neg_s], axis=0)
        .sample(frac=1.0, random_state=random_state)
        .reset_index(drop=True)
    )

    info = {
        "before_total": int(len(df)),
        "before_pos": int(n_pos),
        "before_neg": int(n_neg),
        "after_total": int(len(out)),
        "after_pos": int(minority_n),
        "after_neg": int(minority_n),
    }
    return out, info


def get_first_value(df: pd.DataFrame, col: str, default: str) -> str:
    if col in df.columns and len(df):
        return str(df[col].iloc[0])
    return default


def get_model_metadata(df: pd.DataFrame) -> Dict[str, str]:
    """
    Normalize metadata across Detoxify and generic guardrail outputs.
    """
    dataset_name = get_first_value(df, "dataset_name", "unknown_dataset")
    text_field = get_first_value(df, "text_field", "unknown_text_field")
    split = get_first_value(df, "split", "unknown_split")

    if "safety_model" in df.columns:
        model_family = get_first_value(df, "safety_model", "unknown_model")
    elif "detoxify_model" in df.columns:
        model_family = "detoxify"
    else:
        model_family = "unknown_model"

    if "safety_model_variant" in df.columns:
        model_variant = get_first_value(df, "safety_model_variant", "unknown_variant")
    elif "detoxify_model" in df.columns:
        model_variant = get_first_value(df, "detoxify_model", "unknown_variant")
    else:
        model_variant = "unknown_variant"

    return {
        "dataset_name": dataset_name,
        "text_field": text_field,
        "split": split,
        "safety_model": model_family,
        "safety_model_variant": model_variant,
    }


def build_file_signature(df: pd.DataFrame, path: str | Path) -> Tuple[str, str, str, str]:
    meta = get_model_metadata(df)
    return (
        meta["dataset_name"],
        meta["safety_model"],
        meta["safety_model_variant"],
        meta["text_field"],
    )


def add_common_metadata(row: Dict[str, object], df: pd.DataFrame, input_path: str | Path, source_tag: str) -> None:
    row[f"{source_tag}_path"] = str(input_path)
    meta = get_model_metadata(df)
    row.update(meta)
    # Keep the old column for backward compatibility with existing result readers.
    if meta["safety_model"] == "detoxify":
        row["detoxify_model"] = meta["safety_model_variant"]
    else:
        row["detoxify_model"] = ""


def aggregate_metrics(df: pd.DataFrame, group_cols: Sequence[str], metric_cols: Sequence[str]) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    agg = df.groupby(list(group_cols), dropna=False)[list(metric_cols)].agg(["mean", "std", "min", "max"]).reset_index()
    agg.columns = ["_".join(col).strip("_") if isinstance(col, tuple) else col for col in agg.columns]
    return agg


def score_specs_match(test_specs: List[Dict[str, object]], dev_specs: List[Dict[str, object]]) -> bool:
    test_keys = [(str(s["scoring"]), str(s["mode"]), tuple(s.get("score_cols") or [])) for s in test_specs]
    dev_keys = [(str(s["scoring"]), str(s["mode"]), tuple(s.get("score_cols") or [])) for s in dev_specs]
    return test_keys == dev_keys


def evaluate_single_test_file(
    test_path: str | Path,
    dev_path: Optional[str | Path] = None,
    label_col: Optional[str] = None,
    score_col: Optional[str] = None,
    config: Optional[EvalConfig] = None,
    out_dir: Optional[str | Path] = None,
) -> Dict[str, pd.DataFrame]:
    config = config or EvalConfig()
    effective_score_col = score_col if score_col is not None else config.score_col

    test_path = Path(test_path)
    df_test_all = read_table(test_path)
    schema_info = validate_required_columns(df_test_all, label_col, effective_score_col)
    label_col = str(schema_info["label_col"])
    score_specs = list(schema_info["score_specs"])  # type: ignore[arg-type]

    if dev_path is not None:
        dev_path = Path(dev_path)
        df_dev = read_table(dev_path)
        dev_schema_info = validate_required_columns(df_dev, label_col, effective_score_col)
        dev_label_col = str(dev_schema_info["label_col"])
        dev_score_specs = list(dev_schema_info["score_specs"])  # type: ignore[arg-type]
        if not score_specs_match(score_specs, dev_score_specs):
            raise ValueError(
                "DEV and TEST score specs do not match. "
                f"DEV={dev_score_specs}, TEST={score_specs}"
            )
        df_eval = df_test_all.copy().reset_index(drop=True)
        threshold_source = "provided_dev_file"
    else:
        df_dev, df_eval = stratified_dev_eval_split(
            df=df_test_all,
            label_col=label_col,
            dev_fraction=config.dev_fraction_when_missing,
            random_state=config.split_random_state,
        )
        dev_label_col = label_col
        threshold_source = "split_from_test"

    y_dev = coerce_binary_label(df_dev[dev_label_col], dev_label_col)
    y_eval = coerce_binary_label(df_eval[label_col], label_col)

    raw_rows: List[Dict[str, object]] = []
    balanced_rows: List[Dict[str, object]] = []
    balanced_sampling_rows: List[Dict[str, object]] = []

    for score_spec in score_specs:
        scoring = str(score_spec["scoring"])
        s_dev = compute_scalar_score(df_dev, score_spec)
        s_eval = compute_scalar_score(df_eval, score_spec)
        threshold = select_threshold_for_target_recall(
            y_true_dev=y_dev,
            y_score_dev=s_dev,
            target_recall=config.target_recall,
        )

        raw_row: Dict[str, object] = {
            "evaluation_mode": "raw",
            "scoring": scoring,
            "threshold_source": threshold_source,
            "dev_n": int(len(df_dev)),
            "eval_n": int(len(df_eval)),
            "target_recall": float(config.target_recall),
        }
        add_common_metadata(raw_row, df_eval, test_path, source_tag="test")
        raw_row["dev_path"] = str(dev_path) if dev_path is not None else None
        raw_row.update(compute_threshold_free_metrics(y_eval, s_eval))
        raw_row.update(compute_thresholded_metrics(y_eval, s_eval, threshold))
        raw_rows.append(raw_row)

        for repeat_idx in range(config.n_balance_repeats):
            repeat_seed = config.balance_random_seed + repeat_idx
            df_bal, sample_info = downsample_to_balanced(
                df=df_eval,
                label_col=label_col,
                random_state=repeat_seed,
            )
            y_bal = coerce_binary_label(df_bal[label_col], label_col)
            s_bal = compute_scalar_score(df_bal, score_spec)

            row: Dict[str, object] = {
                "evaluation_mode": "balanced",
                "scoring": scoring,
                "repeat": int(repeat_idx),
                "random_state": int(repeat_seed),
                "threshold_source": threshold_source,
                "dev_n": int(len(df_dev)),
                "eval_n": int(len(df_eval)),
                "target_recall": float(config.target_recall),
            }
            add_common_metadata(row, df_bal, test_path, source_tag="test")
            row["dev_path"] = str(dev_path) if dev_path is not None else None
            row.update(compute_threshold_free_metrics(y_bal, s_bal))
            row.update(compute_thresholded_metrics(y_bal, s_bal, threshold))
            balanced_rows.append(row)

            balanced_sampling_row: Dict[str, object] = {
                "scoring": scoring,
                "repeat": int(repeat_idx),
                "random_state": int(repeat_seed),
            }
            add_common_metadata(balanced_sampling_row, df_bal, test_path, source_tag="test")
            balanced_sampling_row.update(sample_info)
            balanced_sampling_rows.append(balanced_sampling_row)

    raw_df = pd.DataFrame(raw_rows)
    balanced_per_repeat_df = pd.DataFrame(balanced_rows)
    balanced_sampling_df = pd.DataFrame(balanced_sampling_rows)

    balanced_metric_cols = [
        "auroc",
        "auprc",
        "pos_rate",
        "n",
        "threshold",
        "precision",
        "recall",
        "f1",
        "f2",
        "fpr",
        "fnr",
        "specificity",
        "tp",
        "fp",
        "tn",
        "fn",
    ]
    balanced_agg_df = aggregate_metrics(
        balanced_per_repeat_df,
        group_cols=["dataset_name", "safety_model", "safety_model_variant", "text_field", "split", "scoring"],
        metric_cols=balanced_metric_cols,
    )

    outputs = {
        "raw_metrics": raw_df,
        "balanced_per_repeat_metrics": balanced_per_repeat_df,
        "balanced_aggregated_metrics": balanced_agg_df,
        "balanced_sampling_summary": balanced_sampling_df,
    }

    if out_dir is not None and config.save_outputs:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = test_path.stem
        raw_df.to_csv(out_dir / f"{stem}__raw_metrics.csv", index=False)
        balanced_per_repeat_df.to_csv(out_dir / f"{stem}__balanced_per_repeat_metrics.csv", index=False)
        balanced_agg_df.to_csv(out_dir / f"{stem}__balanced_aggregated_metrics.csv", index=False)
        balanced_sampling_df.to_csv(out_dir / f"{stem}__balanced_sampling_summary.csv", index=False)
        with open(out_dir / f"{stem}__run_config.json", "w", encoding="utf-8") as f:
            json.dump(asdict(config), f, indent=2)

    return outputs


def collect_files(path_or_dir: str | Path, file_ext: str = ".parquet") -> List[Path]:
    path = Path(path_or_dir)
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(path.glob(f"*{file_ext}"))
    raise FileNotFoundError(f"Path not found: {path}")


def build_signature_map(
    paths: Sequence[Path],
    label_col: Optional[str] = None,
    score_col: Optional[str] = None,
) -> Dict[Tuple[str, str, str, str], Path]:
    out: Dict[Tuple[str, str, str, str], Path] = {}
    for path in paths:
        df = read_table(path)
        _ = validate_required_columns(df, label_col, score_col)
        signature = build_file_signature(df, path)
        if signature in out:
            raise ValueError(f"Duplicate signature {signature} found for files: {out[signature]} and {path}")
        out[signature] = path
    return out


def evaluate_many(
    test_path_or_dir: str | Path,
    dev_path_or_dir: Optional[str | Path] = None,
    label_col: Optional[str] = None,
    score_col: Optional[str] = None,
    config: Optional[EvalConfig] = None,
    out_dir: Optional[str | Path] = None,
) -> Dict[str, pd.DataFrame]:
    config = config or EvalConfig()
    effective_score_col = score_col if score_col is not None else config.score_col

    test_files = collect_files(test_path_or_dir, config.file_ext)
    if not test_files:
        raise FileNotFoundError(f"No test files found under: {test_path_or_dir}")

    all_raw: List[pd.DataFrame] = []
    all_balanced_repeat: List[pd.DataFrame] = []
    all_balanced_agg: List[pd.DataFrame] = []
    all_balanced_sampling: List[pd.DataFrame] = []

    dev_map: Optional[Dict[Tuple[str, str, str, str], Path]] = None
    if dev_path_or_dir is not None:
        dev_files = collect_files(dev_path_or_dir, config.file_ext)
        if not dev_files:
            raise FileNotFoundError(f"No dev files found under: {dev_path_or_dir}")
        dev_map = build_signature_map(dev_files, label_col=label_col, score_col=effective_score_col)

    for test_file in test_files:
        matched_dev: Optional[Path] = None
        if dev_map is not None:
            test_df = read_table(test_file)
            _ = validate_required_columns(test_df, label_col, effective_score_col)
            signature = build_file_signature(test_df, test_file)
            if signature not in dev_map:
                raise KeyError(
                    f"Could not find matching DEV file for TEST file '{test_file}'. "
                    f"Expected signature={signature}."
                )
            matched_dev = dev_map[signature]

        file_out_dir = None if out_dir is None else Path(out_dir) / test_file.stem
        outputs = evaluate_single_test_file(
            test_path=test_file,
            dev_path=matched_dev,
            label_col=label_col,
            score_col=effective_score_col,
            config=config,
            out_dir=file_out_dir,
        )
        all_raw.append(outputs["raw_metrics"])
        all_balanced_repeat.append(outputs["balanced_per_repeat_metrics"])
        all_balanced_agg.append(outputs["balanced_aggregated_metrics"])
        all_balanced_sampling.append(outputs["balanced_sampling_summary"])

    merged_outputs = {
        "raw_metrics": pd.concat(all_raw, ignore_index=True) if all_raw else pd.DataFrame(),
        "balanced_per_repeat_metrics": pd.concat(all_balanced_repeat, ignore_index=True) if all_balanced_repeat else pd.DataFrame(),
        "balanced_aggregated_metrics": pd.concat(all_balanced_agg, ignore_index=True) if all_balanced_agg else pd.DataFrame(),
        "balanced_sampling_summary": pd.concat(all_balanced_sampling, ignore_index=True) if all_balanced_sampling else pd.DataFrame(),
    }

    if out_dir is not None and config.save_outputs:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, df in merged_outputs.items():
            df.to_csv(out_dir / f"all_files__{name}.csv", index=False)
        with open(out_dir / "all_files__run_config.json", "w", encoding="utf-8") as f:
            json.dump(asdict(config), f, indent=2)

    return merged_outputs


def print_quick_summary(outputs: Dict[str, pd.DataFrame]) -> None:
    raw_df = outputs["raw_metrics"]
    bal_df = outputs["balanced_aggregated_metrics"]

    if not raw_df.empty:
        print("\n=== Raw metrics ===")
        keep = [
            c
            for c in [
                "dataset_name",
                "safety_model",
                "safety_model_variant",
                "text_field",
                "split",
                "scoring",
                "threshold_source",
                "auroc",
                "auprc",
                "precision",
                "recall",
                "f1",
                "f2",
                "fpr",
                "fnr",
                "specificity",
                "threshold",
                "n",
                "pos_rate",
            ]
            if c in raw_df.columns
        ]
        print(raw_df[keep].to_string(index=False))

    if not bal_df.empty:
        print("\n=== Balanced metrics (aggregated over repeats) ===")
        keep = [
            c
            for c in [
                "dataset_name",
                "safety_model",
                "safety_model_variant",
                "text_field",
                "split",
                "scoring",
                "auroc_mean",
                "auroc_std",
                "auprc_mean",
                "auprc_std",
                "precision_mean",
                "precision_std",
                "recall_mean",
                "recall_std",
                "f1_mean",
                "f1_std",
                "f2_mean",
                "f2_std",
                "fpr_mean",
                "fpr_std",
                "fnr_mean",
                "fnr_std",
                "specificity_mean",
                "specificity_std",
                "threshold_mean",
                "threshold_std",
            ]
            if c in bal_df.columns
        ]
        print(bal_df[keep].to_string(index=False))


def make_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Model-agnostic safety detector evaluation pipeline")
    parser.add_argument("--test", required=True, help="Path to a test parquet/csv file, or a directory of such files.")
    parser.add_argument("--dev", default=None, help="Optional path to a dev parquet/csv file, or a directory of such files.")
    parser.add_argument("--label-col", default=None, help="Optional override for label column name.")
    parser.add_argument(
        "--score-col",
        default=None,
        help=(
            "Optional unsafe score column. Higher means more unsafe. "
            "Use this for generic guardrails, e.g. --score-col guard_unsafe_score or --score-col unsafe_score."
        ),
    )
    parser.add_argument("--out-dir", default="safety_eval_outputs", help="Directory to save CSV outputs.")
    parser.add_argument("--target-recall", type=float, default=0.95, help="Target recall used to choose threshold on DEV.")
    parser.add_argument("--dev-fraction-when-missing", type=float, default=0.20, help="Fraction used as DEV when no external dev file is provided.")
    parser.add_argument("--split-random-state", type=int, default=42, help="Random seed for test->dev/eval split when DEV is missing.")
    parser.add_argument("--n-balance-repeats", type=int, default=30, help="Number of balanced repeated samplings.")
    parser.add_argument("--balance-random-seed", type=int, default=1234, help="Base random seed for balanced repeated sampling.")
    parser.add_argument("--file-ext", default=".parquet", help="File extension to search inside directories, e.g. .parquet")
    parser.add_argument("--no-save", action="store_true", help="Do not save CSV outputs.")
    return parser


def main() -> None:
    parser = make_arg_parser()
    args = parser.parse_args()

    config = EvalConfig(
        dev_fraction_when_missing=args.dev_fraction_when_missing,
        split_random_state=args.split_random_state,
        target_recall=args.target_recall,
        n_balance_repeats=args.n_balance_repeats,
        balance_random_seed=args.balance_random_seed,
        file_ext=args.file_ext,
        save_outputs=not args.no_save,
        score_col=args.score_col,
    )

    outputs = evaluate_many(
        test_path_or_dir=args.test,
        dev_path_or_dir=args.dev,
        label_col=args.label_col,
        score_col=args.score_col,
        config=config,
        out_dir=args.out_dir,
    )
    print_quick_summary(outputs)


if __name__ == "__main__":
    main()
