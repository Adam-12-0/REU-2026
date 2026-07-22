# -*- coding: utf-8 -*-

"""Majority-vote fusion for compiled SC, LG, and MA predictions.

This script consumes the output of compile.py and mirrors lf.py's output
interface, but replaces logistic regression with an unweighted vote over
sc_pred, lg_pred, and ma_pred.
"""

import argparse
from pathlib import Path

import pandas as pd


PREDICTION_COLUMNS = ["sc_pred", "lg_pred", "ma_pred"]
POSITIVE_LABELS = {"1", "true", "t", "yes", "y", "gold", "specialized"}
NEGATIVE_LABELS = {
    "0",
    "false",
    "f",
    "no",
    "n",
    "not_gold",
    "non_gold",
    "general",
    "",
}
SINGLE_TYPES = {"word", "single_word", "is_single_word"}
MULTI_TYPES = {"phrase", "multiword", "multi_word", "is_multiword"}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Fuse compiled SC, LG, and MA predictions by vote and select the "
            "vote threshold with the highest F1."
        )
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--all-output", type=Path, required=True)
    parser.add_argument("--single-output", type=Path, required=True)
    parser.add_argument("--multi-output", type=Path, required=True)
    parser.add_argument("--thresholds-output", type=Path, required=True)
    parser.add_argument("--metrics-output", type=Path, required=True)
    parser.add_argument(
        "--vote-threshold",
        type=int,
        choices=range(1, len(PREDICTION_COLUMNS) + 1),
        default=None,
        help=(
            "Optional fixed number of required positive votes. If omitted, "
            "the threshold is selected by F1 on the input labels. Use 2 for "
            "a strict majority of the three methods."
        ),
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


def apply_vote_threshold(df, threshold):
    output = df.copy()
    output["vote_pred"] = (output["vote_count"] >= int(threshold)).astype(int)
    return apply_phrase_picker(output)


def apply_phrase_picker(df):
    """Prefer high-scoring, non-overlapping phrases over component words."""
    output = df.copy()
    output["_numeric_token_pos"] = pd.to_numeric(
        output.get("token_pos"), errors="coerce"
    )

    type_norm = output["cand_type"].map(normalize_candidate_type)
    phrase_mask = type_norm.isin(MULTI_TYPES) & output["vote_pred"].eq(1)
    phrase_df = output.loc[phrase_mask].copy()

    if phrase_df.empty:
        return output.drop(columns=["_numeric_token_pos"])

    if "component_token_positions" not in output.columns:
        # The compiler should normally preserve this span information. Without
        # it, retain vote predictions rather than guessing phrase membership.
        return output.drop(columns=["_numeric_token_pos"])

    phrase_df["_length"] = phrase_df["component_token_positions"].apply(
        lambda value: len(parse_component_positions(value))
    )
    claimed_positions = set()

    for phrase_index, phrase_row in phrase_df.sort_values(
        ["vote_score", "_length"],
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
            output.loc[phrase_index, "vote_pred"] = 0
            continue

        word_mask = (
            output["sentence_id"].astype(str).eq(sentence_id)
            & type_norm.isin(SINGLE_TYPES)
            & output["_numeric_token_pos"].isin(positions)
            & output["vote_pred"].eq(1)
        )
        output.loc[word_mask, "vote_pred"] = 0
        claimed_positions.update(
            (sentence_id, position) for position in positions
        )

    return output.drop(columns=["_numeric_token_pos"])


def evaluate_thresholds(df):
    rows = []
    for threshold in range(1, len(PREDICTION_COLUMNS) + 1):
        predicted = apply_vote_threshold(df, threshold)
        metrics = calculate_metrics(
            predicted["is_gold_binary"], predicted["vote_pred"]
        )
        rows.append(
            {
                "threshold": threshold,
                **metrics,
                "is_best": 0,
                "selected_for_prediction": 0,
            }
        )
    return pd.DataFrame(rows)


def select_threshold(thresholds_df, fixed_threshold=None):
    if fixed_threshold is not None:
        return int(fixed_threshold)

    best = thresholds_df.sort_values(
        ["f1", "precision", "recall", "threshold"],
        ascending=[False, False, False, False],
        kind="mergesort",
    ).iloc[0]
    return int(best["threshold"])


def metrics_row(cand_type, df):
    metrics = calculate_metrics(df["is_gold_binary"], df["vote_pred"])
    return {"cand_type": cand_type, **metrics}


def write_metrics(df, output_path):
    type_norm = df["cand_type"].map(normalize_candidate_type)
    rows = []

    single_df = df.loc[type_norm.isin(SINGLE_TYPES)]
    multi_df = df.loc[type_norm.isin(MULTI_TYPES)]
    rows.append(metrics_row("is_single_word", single_df))
    rows.append(metrics_row("is_multiword", multi_df))
    rows.append(metrics_row("all", df))
    pd.DataFrame(rows).to_csv(output_path, index=False)


def ensure_parent_directories(paths):
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)


def main():
    args = parse_args()
    ensure_parent_directories(
        [
            args.all_output,
            args.single_output,
            args.multi_output,
            args.thresholds_output,
            args.metrics_output,
        ]
    )

    df = pd.read_csv(args.input)
    require_columns(
        df,
        [
            "sentence_id",
            "cand_type",
            "is_gold",
            *PREDICTION_COLUMNS,
        ],
        args.input,
    )

    working = df.copy()
    working["is_gold_binary"] = working["is_gold"].map(
        lambda value: to_binary(value, "is_gold")
    )
    for column in PREDICTION_COLUMNS:
        working[column] = working[column].map(
            lambda value, name=column: to_binary(value, name)
        )

    working["vote_count"] = working[PREDICTION_COLUMNS].sum(axis=1)
    working["vote_score"] = working["vote_count"] / len(PREDICTION_COLUMNS)

    thresholds_df = evaluate_thresholds(working)
    selected_threshold = select_threshold(
        thresholds_df, args.vote_threshold
    )
    selected_mask = thresholds_df["threshold"].eq(selected_threshold)
    thresholds_df.loc[
        selected_mask, ["is_best", "selected_for_prediction"]
    ] = 1
    thresholds_df.to_csv(args.thresholds_output, index=False)

    final_df = apply_vote_threshold(working, selected_threshold)
    final_df["final_score"] = final_df["vote_score"]
    final_df["final_pred"] = final_df["vote_pred"]
    final_df = final_df.drop(columns=["is_gold_binary"])

    type_norm = final_df["cand_type"].map(normalize_candidate_type)
    single_df = final_df.loc[type_norm.isin(SINGLE_TYPES)].copy()
    multi_df = final_df.loc[type_norm.isin(MULTI_TYPES)].copy()

    final_df.to_csv(args.all_output, index=False)
    single_df.to_csv(args.single_output, index=False)
    multi_df.to_csv(args.multi_output, index=False)

    metrics_input = final_df.copy()
    metrics_input["is_gold_binary"] = metrics_input["is_gold"].map(
        lambda value: to_binary(value, "is_gold")
    )
    write_metrics(metrics_input, args.metrics_output)

    best_row = thresholds_df.loc[selected_mask].iloc[0]
    print(
        f"Selected vote threshold: {selected_threshold}/"
        f"{len(PREDICTION_COLUMNS)} | "
        f"precision={best_row['precision']:.4f} | "
        f"recall={best_row['recall']:.4f} | "
        f"f1={best_row['f1']:.4f}",
        flush=True,
    )
    print(f"Wrote all predictions to {args.all_output}", flush=True)
    print(f"Wrote single-word predictions to {args.single_output}", flush=True)
    print(f"Wrote multiword predictions to {args.multi_output}", flush=True)
    print(f"Wrote threshold metrics to {args.thresholds_output}", flush=True)
    print(f"Wrote final metrics to {args.metrics_output}", flush=True)


if __name__ == "__main__":
    main()
