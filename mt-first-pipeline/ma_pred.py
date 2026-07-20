# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import pandas as pd
import regex as re


WORD_RE = re.compile(r"[\p{L}\p{M}]+(?:['\u2019-][\p{L}\p{M}]+)?|[\p{N}]+")


def parse_args():
    parser = argparse.ArgumentParser(description="Apply selected MA threshold and write final predictions.")
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--thresholds", type=Path, required=True)
    parser.add_argument("--single-output", type=Path, required=True)
    parser.add_argument("--multi-output", type=Path, required=True)
    parser.add_argument("--all-output", type=Path, required=True)
    parser.add_argument("--metrics-output", type=Path, required=True)
    return parser.parse_args()


def require_columns(df, columns, path):
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")


def to_binary(value):
    if pd.isna(value):
        return 0
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "gold", "specialized"}:
        return 1
    if normalized in {"0", "false", "f", "no", "n", "not_gold", "non_gold", "general", ""}:
        return 0
    return int(float(normalized) > 0)


def load_selected_threshold(path):
    thresholds = pd.read_csv(path)
    require_columns(thresholds, ["threshold", "selected_for_prediction"], path)
    selected = thresholds[thresholds["selected_for_prediction"].astype(int).eq(1)]
    if selected.empty:
        selected = thresholds.sort_values(["precision", "f1", "recall", "threshold"], ascending=[False, False, False, True]).head(1)
    if selected.empty:
        raise ValueError(f"No selected threshold found in {path}")
    return float(selected.iloc[0]["threshold"])


def tokenize_words(text):
    return [match.group(0) for match in WORD_RE.finditer(str(text))]


def parse_component_positions(value):
    positions = []
    for item in str(value).split("|"):
        if item == "":
            continue
        try:
            positions.append(int(item))
        except ValueError:
            continue
    return positions


def component_positions(row):
    if str(row.get("cand_type", "")).lower() != "is_multiword":
        return ""
    if "component_token_positions" in row and pd.notna(row["component_token_positions"]):
        existing = str(row["component_token_positions"])
        if existing.strip():
            return existing
    try:
        start = int(float(row.get("token_pos", "")))
    except (TypeError, ValueError):
        return ""
    if "end_token_pos" in row and pd.notna(row["end_token_pos"]):
        try:
            end = int(float(row["end_token_pos"]))
            return "|".join(str(pos) for pos in range(start, end + 1))
        except (TypeError, ValueError):
            pass
    token_count = len(tokenize_words(row.get("candidate", "")))
    if token_count < 2:
        token_count = len(tokenize_words(row.get("cand_norm", "")))
    if token_count < 2:
        return ""
    return "|".join(str(pos) for pos in range(start, start + token_count))


def apply_phrase_picker(pred_df):
    final_df = pred_df.copy()
    final_df["_numeric_token_pos"] = pd.to_numeric(final_df["token_pos"], errors="coerce")
    if "component_token_positions" not in final_df.columns:
        final_df["component_token_positions"] = ""
    final_df["component_token_positions"] = final_df.apply(component_positions, axis=1)

    phrase_df = final_df[
        final_df["cand_type"].astype(str).str.lower().eq("is_multiword")
        & final_df["ma_pred"].astype(int).eq(1)
    ].copy()
    if phrase_df.empty:
        return final_df.drop(columns=["_numeric_token_pos"])

    phrase_df["_length"] = phrase_df["component_token_positions"].apply(lambda value: len(parse_component_positions(value)))
    claimed_positions = set()
    for phrase_index, phrase_row in phrase_df.sort_values(["ma_score", "_length"], ascending=[False, False], kind="mergesort").iterrows():
        sentence_id = str(phrase_row["sentence_id"])
        positions = parse_component_positions(phrase_row["component_token_positions"])
        if not positions or any((sentence_id, position) in claimed_positions for position in positions):
            final_df.loc[phrase_index, "ma_pred"] = 0
            continue
        word_mask = (
            final_df["sentence_id"].astype(str).eq(sentence_id)
            & final_df["cand_type"].astype(str).str.lower().eq("is_single_word")
            & final_df["_numeric_token_pos"].isin(positions)
        )
        max_word_score = final_df.loc[word_mask, "ma_score"].astype(float).max()
        if pd.notna(max_word_score) and float(phrase_row["ma_score"]) > float(max_word_score):
            final_df.loc[phrase_index, "ma_pred"] = 1
            final_df.loc[word_mask, "ma_pred"] = 0
            for position in positions:
                claimed_positions.add((sentence_id, position))
        else:
            final_df.loc[phrase_index, "ma_pred"] = 0
    return final_df.drop(columns=["_numeric_token_pos"])


def calculate_counts(y_true, y_pred):
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return tp, fp, fn


def calculate_metrics(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def metric_row(cand_type, df):
    y_true = df["is_gold"].astype(int)
    y_pred = df["ma_pred"].astype(int)
    tp, fp, fn = calculate_counts(y_true, y_pred)
    precision, recall, f1 = calculate_metrics(tp, fp, fn)
    return {"cand_type": cand_type, "precision": precision, "recall": recall, "f1": f1}


def write_metrics(df, output_path):
    rows = [metric_row(cand_type, group) for cand_type, group in df.groupby("cand_type", sort=True)]
    rows.append(metric_row("all", df))
    pd.DataFrame(rows).to_csv(output_path, index=False)


def output_columns(df):
    preferred = ["sentence_id", "sentence", "candidate", "cand_norm", "cand_type", "is_gold", "token_pos", "end_token_pos", "component_token_positions", "mt_pred", "ma_score", "ma_pred"]
    columns = [column for column in preferred if column in df.columns]
    columns.extend(column for column in df.columns if column not in columns)
    return columns


def main():
    args = parse_args()
    threshold = load_selected_threshold(args.thresholds)
    scores_df = pd.read_csv(args.scores)
    if "candidate_type" in scores_df.columns and "cand_type" not in scores_df.columns:
        scores_df = scores_df.rename(columns={"candidate_type": "cand_type"})
    require_columns(scores_df, ["sentence_id", "sentence", "candidate", "cand_norm", "cand_type", "is_gold", "mt_pred", "mean_similarity"], args.scores)

    output_df = scores_df.copy()
    if "ma_score" not in output_df.columns:
        output_df["ma_score"] = 1.0 - output_df["mean_similarity"].astype(float)
    output_df["ma_pred"] = 0
    score_mask = output_df["mt_pred"].apply(to_binary).eq(1) & output_df["ma_score"].notna()
    output_df.loc[score_mask, "ma_pred"] = (output_df.loc[score_mask, "ma_score"] <= threshold).astype(int)
    output_df = apply_phrase_picker(output_df)
    output_df = output_df[output_columns(output_df)]

    single_df = output_df[output_df["cand_type"].astype(str).str.lower().eq("is_single_word")].copy()
    multi_df = output_df[output_df["cand_type"].astype(str).str.lower().eq("is_multiword")].copy()
    single_df.to_csv(args.single_output, index=False)
    multi_df.to_csv(args.multi_output, index=False)
    output_df.to_csv(args.all_output, index=False)
    write_metrics(output_df, args.metrics_output)

    print(f"Selected threshold: {threshold}", flush=True)
    print("Applied rule: mean_similarity <= threshold", flush=True)
    print(f"Wrote single-word predictions to {args.single_output}", flush=True)
    print(f"Wrote multiword predictions to {args.multi_output}", flush=True)
    print(f"Wrote all predictions to {args.all_output}", flush=True)
    print(f"Wrote metrics to {args.metrics_output}", flush=True)


if __name__ == "__main__":
    main()
