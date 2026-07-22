# -*- coding: utf-8 -*-

"""Fixed-weight fusion for the validated output of compile.py."""

import argparse
import math
from pathlib import Path

import pandas as pd


WEIGHTS = {"sc_pred": 0.20, "lg_pred": 0.40, "ma_pred": 0.40}
POSITIVE_LABELS = {"1", "true", "t", "yes", "y", "gold", "specialized"}
NEGATIVE_LABELS = {
    "0", "false", "f", "no", "n", "not_gold", "non_gold", "general", ""
}
SINGLE_TYPES = {"word", "single_word", "is_single_word"}
MULTI_TYPES = {"phrase", "multiword", "multi_word", "is_multiword"}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Fuse compiled SC, LG, and MA predictions using fixed weights "
            "SC=0.20, LG=0.40, and MA=0.40."
        )
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--all-output", type=Path, required=True)
    parser.add_argument("--single-output", type=Path, required=True)
    parser.add_argument("--multi-output", type=Path, required=True)
    parser.add_argument("--thresholds-output", type=Path, required=True)
    parser.add_argument("--weights-output", type=Path, required=True)
    parser.add_argument("--metrics-output", type=Path, required=True)
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Fixed decision threshold. If omitted, select it by highest F1.",
    )
    return parser.parse_args()


def require_columns(df, columns, path):
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")


def to_binary(value, column_name):
    if pd.isna(value):
        return 0
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value > 0)
    normalized = str(value).strip().lower()
    if normalized in POSITIVE_LABELS:
        return 1
    if normalized in NEGATIVE_LABELS:
        return 0
    try:
        return int(float(normalized) > 0)
    except ValueError as exc:
        raise ValueError(
            f"Cannot parse {column_name} value as binary: {value!r}"
        ) from exc


def normalize_candidate_type(value):
    return str(value).strip().lower()


def parse_component_positions(value):
    if pd.isna(value):
        return []
    positions = []
    for item in str(value).split("|"):
        item = item.strip()
        if not item:
            continue
        try:
            positions.append(int(float(item)))
        except ValueError:
            continue
    return positions


def calculate_metrics(y_true, y_pred):
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "support": int((y_true == 1).sum()),
        "predicted_count": int((y_pred == 1).sum()),
        "candidate_count": int(len(y_true)),
    }


def apply_phrase_picker(df):
    output = df.copy()
    output["_numeric_token_pos"] = pd.to_numeric(
        output.get("token_pos"), errors="coerce"
    )
    type_norm = output["cand_type"].map(normalize_candidate_type)
    phrase_df = output.loc[
        type_norm.isin(MULTI_TYPES) & output["weighted_pred"].eq(1)
    ].copy()

    if phrase_df.empty or "component_token_positions" not in output.columns:
        return output.drop(columns=["_numeric_token_pos"])

    phrase_df["_length"] = phrase_df["component_token_positions"].apply(
        lambda value: len(parse_component_positions(value))
    )
    claimed_positions = set()

    for phrase_index, phrase_row in phrase_df.sort_values(
        ["weighted_score", "_length"],
        ascending=[False, False],
        kind="mergesort",
    ).iterrows():
        sentence_id = str(phrase_row["sentence_id"])
        positions = parse_component_positions(
            phrase_row["component_token_positions"]
        )
        if not positions or any(
            (sentence_id, position) in claimed_positions
            for position in positions
        ):
            output.loc[phrase_index, "weighted_pred"] = 0
            continue

        word_mask = (
            output["sentence_id"].astype(str).eq(sentence_id)
            & type_norm.isin(SINGLE_TYPES)
            & output["_numeric_token_pos"].isin(positions)
            & output["weighted_pred"].eq(1)
        )
        output.loc[word_mask, "weighted_pred"] = 0
        claimed_positions.update(
            (sentence_id, position) for position in positions
        )

    return output.drop(columns=["_numeric_token_pos"])


def apply_threshold(df, threshold):
    output = df.copy()
    output["weighted_pred"] = (
        output["weighted_score"] >= float(threshold)
    ).astype(int)
    return apply_phrase_picker(output)


def candidate_thresholds(scores):
    values = sorted(pd.to_numeric(scores, errors="coerce").dropna().unique())
    if not values:
        raise ValueError("No finite weighted scores were available.")
    return values + [math.nextafter(float(values[-1]), math.inf)]


def evaluate_thresholds(df):
    rows = []
    for threshold in candidate_thresholds(df["weighted_score"]):
        predicted = apply_threshold(df, threshold)
        metrics = calculate_metrics(
            predicted["is_gold_binary"], predicted["weighted_pred"]
        )
        rows.append(
            {
                "threshold": float(threshold),
                **metrics,
                "is_best": 0,
                "selected_for_prediction": 0,
            }
        )
    return pd.DataFrame(rows)


def choose_threshold(thresholds_df, fixed_threshold):
    if fixed_threshold is not None:
        return float(fixed_threshold)
    best = thresholds_df.sort_values(
        ["f1", "precision", "recall", "threshold"],
        ascending=[False, False, False, False],
        kind="mergesort",
    ).iloc[0]
    return float(best["threshold"])


def metrics_row(cand_type, df):
    return {
        "cand_type": cand_type,
        **calculate_metrics(df["is_gold_binary"], df["weighted_pred"]),
    }


def write_metrics(df, path):
    type_norm = df["cand_type"].map(normalize_candidate_type)
    rows = [
        metrics_row("is_single_word", df.loc[type_norm.isin(SINGLE_TYPES)]),
        metrics_row("is_multiword", df.loc[type_norm.isin(MULTI_TYPES)]),
        metrics_row("all", df),
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


def ensure_parent_directories(paths):
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)


def main():
    args = parse_args()
    ensure_parent_directories(
        [args.all_output, args.single_output, args.multi_output,
         args.thresholds_output, args.weights_output, args.metrics_output]
    )

    df = pd.read_csv(args.input)
    require_columns(
        df,
        ["sentence_id", "cand_type", "is_gold", *WEIGHTS],
        args.input,
    )
    working = df.copy()
    working["is_gold_binary"] = working["is_gold"].map(
        lambda value: to_binary(value, "is_gold")
    )
    for column in WEIGHTS:
        working[column] = working[column].map(
            lambda value, name=column: to_binary(value, name)
        )

    working["weighted_score"] = sum(
        weight * working[column] for column, weight in WEIGHTS.items()
    )
    thresholds_df = evaluate_thresholds(working)
    threshold = choose_threshold(thresholds_df, args.threshold)

    if not thresholds_df["threshold"].eq(threshold).any():
        fixed_result = apply_threshold(working, threshold)
        metrics = calculate_metrics(
            fixed_result["is_gold_binary"], fixed_result["weighted_pred"]
        )
        thresholds_df = pd.concat(
            [thresholds_df, pd.DataFrame([{
                "threshold": threshold, **metrics,
                "is_best": 0, "selected_for_prediction": 0,
            }])],
            ignore_index=True,
        )

    selected_mask = thresholds_df["threshold"].eq(threshold)
    thresholds_df.loc[
        selected_mask, ["is_best", "selected_for_prediction"]
    ] = 1
    thresholds_df.to_csv(args.thresholds_output, index=False)

    pd.DataFrame(
        [{"prediction_column": column, "weight": weight}
         for column, weight in WEIGHTS.items()]
    ).to_csv(args.weights_output, index=False)

    final_df = apply_threshold(working, threshold)
    final_df["final_score"] = final_df["weighted_score"]
    final_df["final_pred"] = final_df["weighted_pred"]
    final_df = final_df.drop(columns=["is_gold_binary"])

    type_norm = final_df["cand_type"].map(normalize_candidate_type)
    single_df = final_df.loc[type_norm.isin(SINGLE_TYPES)].copy()
    multi_df = final_df.loc[type_norm.isin(MULTI_TYPES)].copy()
    final_df.to_csv(args.all_output, index=False)
    single_df.to_csv(args.single_output, index=False)
    multi_df.to_csv(args.multi_output, index=False)

    metrics_df = final_df.copy()
    metrics_df["is_gold_binary"] = metrics_df["is_gold"].map(
        lambda value: to_binary(value, "is_gold")
    )
    write_metrics(metrics_df, args.metrics_output)

    best = thresholds_df.loc[selected_mask].iloc[0]
    print(
        f"Selected weighted threshold: {threshold:.12g} | "
        f"precision={best['precision']:.4f} | "
        f"recall={best['recall']:.4f} | f1={best['f1']:.4f}",
        flush=True,
    )
    print(f"Wrote all predictions to {args.all_output}", flush=True)
    print(f"Wrote single-word predictions to {args.single_output}", flush=True)
    print(f"Wrote multiword predictions to {args.multi_output}", flush=True)
    print(f"Wrote threshold metrics to {args.thresholds_output}", flush=True)
    print(f"Wrote weights to {args.weights_output}", flush=True)
    print(f"Wrote final metrics to {args.metrics_output}", flush=True)


if __name__ == "__main__":
    main()
