# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import regex as re
from sklearn.linear_model import LogisticRegression


PRED_COLUMNS = ["sc_pred", "lg_pred", "ma_pred"]
WORD_RE = re.compile(r"[\p{L}\p{M}]+(?:['\u2019-][\p{L}\p{M}]+)?|[\p{N}]+")


def parse_args():
    parser = argparse.ArgumentParser(description="Train late-fusion logistic regression and write predictions.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--all-output", type=Path, required=True)
    parser.add_argument("--single-output", type=Path, required=True)
    parser.add_argument("--multi-output", type=Path, required=True)
    parser.add_argument("--thresholds-output", type=Path, required=True)
    parser.add_argument("--weights-output", type=Path, required=True)
    parser.add_argument("--metrics-output", type=Path, required=True)
    parser.add_argument("--thresholds-input", type=Path, default=None,
                        help="Optional calibration threshold table to apply.")
    parser.add_argument("--weights-input", type=Path, default=None,
                        help="Optional calibrated logistic-regression weights to apply.")
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


def tokenize_words(text):
    return [match.group(0) for match in WORD_RE.finditer(str(text))]


def calculate_counts(y_true, y_pred):
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    return tp, fp, fn, tn


def calculate_metrics(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


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
    existing = str(row.get("component_token_positions", ""))
    if existing and existing.lower() != "nan":
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
    count = len(tokenize_words(row.get("candidate", "")))
    if count < 2:
        count = len(tokenize_words(row.get("cand_norm", "")))
    if count < 2:
        return ""
    return "|".join(str(pos) for pos in range(start, start + count))


def apply_phrase_picker(pred_df):
    final_df = pred_df.copy()
    final_df["_numeric_token_pos"] = pd.to_numeric(final_df.get("token_pos"), errors="coerce")
    if "component_token_positions" not in final_df.columns:
        final_df["component_token_positions"] = ""
    final_df["component_token_positions"] = final_df.apply(component_positions, axis=1)

    phrase_df = final_df[
        final_df["cand_type"].astype(str).str.lower().eq("is_multiword")
        & final_df["lf_pred"].astype(int).eq(1)
    ].copy()
    if phrase_df.empty:
        return final_df.drop(columns=["_numeric_token_pos"])

    phrase_df["_length"] = phrase_df["component_token_positions"].apply(lambda value: len(parse_component_positions(value)))
    claimed_positions = set()
    for phrase_index, phrase_row in phrase_df.sort_values(["lf_score", "_length"], ascending=[False, False], kind="mergesort").iterrows():
        sentence_id = str(phrase_row["sentence_id"])
        positions = parse_component_positions(phrase_row["component_token_positions"])
        if not positions or any((sentence_id, position) in claimed_positions for position in positions):
            final_df.loc[phrase_index, "lf_pred"] = 0
            continue
        word_mask = (
            final_df["sentence_id"].astype(str).eq(sentence_id)
            & final_df["cand_type"].astype(str).str.lower().eq("is_single_word")
            & final_df["_numeric_token_pos"].isin(positions)
        )
        max_word_score = final_df.loc[word_mask, "lf_score"].astype(float).max()
        if pd.notna(max_word_score) and float(phrase_row["lf_score"]) > float(max_word_score):
            final_df.loc[phrase_index, "lf_pred"] = 1
            final_df.loc[word_mask, "lf_pred"] = 0
            for position in positions:
                claimed_positions.add((sentence_id, position))
        else:
            final_df.loc[phrase_index, "lf_pred"] = 0
    return final_df.drop(columns=["_numeric_token_pos"])


def threshold_rows(base_df):
    rows = []
    y_true = base_df["is_gold"].astype(int)
    for threshold in sorted(base_df["lf_score"].dropna().unique()):
        pred_df = base_df.copy()
        pred_df["lf_pred"] = (pred_df["lf_score"].astype(float) >= float(threshold)).astype(int)
        pred_df = apply_phrase_picker(pred_df)
        y_pred = pred_df["lf_pred"].astype(int)
        tp, fp, fn, tn = calculate_counts(y_true, y_pred)
        precision, recall, f1 = calculate_metrics(tp, fp, fn)
        rows.append(
            {
                "threshold": threshold,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "predicted_count": int(y_pred.sum()),
                "gold_count": int(y_true.sum()),
                "candidate_count": int(len(y_true)),
                "is_best": 0,
            }
        )
    threshold_df = pd.DataFrame(rows)
    if not threshold_df.empty:
        best_idx = threshold_df.sort_values(["f1", "precision", "recall", "threshold"], ascending=[False, False, False, True]).index[0]
        threshold_df.loc[best_idx, "is_best"] = 1
    return threshold_df


def write_metrics(df, output_path):
    rows = []
    for cand_type, group in df.groupby("cand_type", sort=True):
        tp, fp, fn, _tn = calculate_counts(group["is_gold"].astype(int), group["lf_pred"].astype(int))
        precision, recall, f1 = calculate_metrics(tp, fp, fn)
        rows.append({"cand_type": cand_type, "precision": precision, "recall": recall, "f1": f1})
    tp, fp, fn, _tn = calculate_counts(df["is_gold"].astype(int), df["lf_pred"].astype(int))
    precision, recall, f1 = calculate_metrics(tp, fp, fn)
    rows.append({"cand_type": "all", "precision": precision, "recall": recall, "f1": f1})
    pd.DataFrame(rows).to_csv(output_path, index=False)


def write_weights(model, output_path):
    weights_df = pd.DataFrame({"feature": PRED_COLUMNS, "weight": model.coef_[0], "abs_weight": abs(model.coef_[0])})
    intercept = pd.DataFrame({"feature": ["intercept"], "weight": [model.intercept_[0]], "abs_weight": [abs(model.intercept_[0])]})
    pd.concat([weights_df.sort_values("abs_weight", ascending=False), intercept], ignore_index=True).to_csv(output_path, index=False)


def load_fixed_weights(path):
    weights_df = pd.read_csv(path)
    require_columns(weights_df, ["feature", "weight"], path)
    weights = weights_df.set_index("feature")["weight"]
    missing = [feature for feature in [*PRED_COLUMNS, "intercept"] if feature not in weights.index]
    if missing:
        raise ValueError(f"{path} is missing calibrated weights for: {missing}")
    return {feature: float(weights[feature]) for feature in PRED_COLUMNS}, float(weights["intercept"]), weights_df


def load_selected_threshold(path):
    thresholds_df = pd.read_csv(path)
    require_columns(thresholds_df, ["threshold"], path)
    for column in ("selected_for_prediction", "is_best"):
        if column in thresholds_df.columns:
            selected = thresholds_df[thresholds_df[column].apply(to_binary).eq(1)]
            if not selected.empty:
                return float(selected.iloc[0]["threshold"])
    if thresholds_df.empty:
        raise ValueError(f"No threshold found in {path}")
    return float(thresholds_df.sort_values(
        ["f1", "precision", "recall", "threshold"],
        ascending=[False, False, False, True],
    ).iloc[0]["threshold"])


def main():
    args = parse_args()
    df = pd.read_csv(args.input)
    require_columns(df, ["is_gold", "cand_type", *PRED_COLUMNS], args.input)
    df = df.copy()
    for column in PRED_COLUMNS:
        df[column] = df[column].apply(to_binary)
    df["is_gold"] = df["is_gold"].apply(to_binary)
    if args.weights_input is None and df["is_gold"].nunique() < 2:
        raise ValueError("Need both positive and negative gold labels to train logistic regression.")

    x = df[PRED_COLUMNS]
    if args.weights_input is not None:
        fixed_weights, intercept, calibrated_weights_df = load_fixed_weights(args.weights_input)
        linear_score = sum(x[feature].astype(float) * weight for feature, weight in fixed_weights.items()) + intercept
        df["lf_score"] = 1.0 / (1.0 + np.exp(-linear_score))
        calibrated_weights_df.to_csv(args.weights_output, index=False)
    else:
        y = df["is_gold"].astype(int)
        model = LogisticRegression(max_iter=1000, class_weight="balanced")
        model.fit(x, y)
        df["lf_score"] = model.predict_proba(x)[:, 1]

    thresholds_df = threshold_rows(df)
    thresholds_df.to_csv(args.thresholds_output, index=False)
    best = thresholds_df[thresholds_df["is_best"].eq(1)].iloc[0]
    applied_threshold = load_selected_threshold(args.thresholds_input) if args.thresholds_input is not None else float(best["threshold"])
    thresholds_df["selected_for_prediction"] = thresholds_df["threshold"].eq(applied_threshold).astype(int)
    thresholds_df.to_csv(args.thresholds_output, index=False)
    df["lf_pred"] = (df["lf_score"] >= applied_threshold).astype(int)
    df = apply_phrase_picker(df)

    single_df = df[df["cand_type"].astype(str).str.lower().eq("is_single_word")].copy()
    multi_df = df[df["cand_type"].astype(str).str.lower().eq("is_multiword")].copy()
    df.to_csv(args.all_output, index=False)
    single_df.to_csv(args.single_output, index=False)
    multi_df.to_csv(args.multi_output, index=False)
    if args.weights_input is None:
        write_weights(model, args.weights_output)
    write_metrics(df, args.metrics_output)
    print(f"Wrote LF predictions to {args.all_output}", flush=True)


if __name__ == "__main__":
    main()
