"""Receive the vote total count from scoring candidates and find the metrics of each threshold of votes.
Also, determine the best threshold by f1 score."""

import argparse
from pathlib import Path
import unicodedata

import regex as re
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

SCORE_COLUMNS = [
    "sentence_id",
    "candidate",
    "cand_norm",
    "vote_score"
]
KEY_COLUMNS = [
    "sentence_id",
    "special_term",
    "term_type"
]

# Evaluate one threshold
def calculate_threshold_metrics(preds, gold, threshold):
    selected = preds[preds["vote_score"] >= threshold]

    predicted_terms = {(row.sentence_id, row.cand_norm) for row in selected.itertuples(index=False)}

    universe = sorted(predicted_terms | gold, key=str)

    y_true = [int(identity in gold) for identity in universe]
    y_pred = [int(identity in predicted_terms) for identity in universe]

    # Save metrics for precision, recall, f1
    if universe:
        precision, recall, f1, _ = (precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0))
    else:
        precision = 0.0
        recall = 0.0
        f1 = 0.0

    # Save TP, FP, FN counts
    true_positive = len(predicted_terms & gold)
    false_positive = len(predicted_terms - gold)
    false_negative = len(gold - predicted_terms)

    return {
        "threshold": float(threshold),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "predicted_positive": len(predicted_terms),
        "gold_positive": len(gold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


# Select thresholds to test on
def candidate_thresholds(count):
    if count < 1:
        raise ValueError("Threshold count must be at least 1.")
    return np.linspace(0.0, 1.0, count)

# Helper function to normalize term
def normalize_word(term):
    term = unicodedata.normalize("NFKC", str(term))
    term = re.sub(r"\s+", " ", term)
    return term.strip().casefold()


# Helper function to check if required columns exist in input
def require_columns(df, reqs, path):
    missing = []
    for col in reqs:
        if col not in df.columns:
            missing.append(col)
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")


def load_gold_terms(path):
    df = pd.read_csv(path, dtype={"sentence_id":"string"})
    require_columns(df, KEY_COLUMNS, path)

    # Remove rows with missing info
    df = df.dropna(subset=["sentence_id", "special_term", "term_type"]).copy()
    
    # Normalize data
    df["sentence_id"] = df["sentence_id"].str.strip()
    df["special_term"] = (df["special_term"].apply(normalize_word))

    df = df[df["sentence_id"].ne("") & df["special_term"].ne("") & df["term_type"].eq("single_word")].copy()

    return {(row.sentence_id, row.special_term) for row in df.itertuples(index=False)}


def load_candidate_scores(path):
    # Load vote scores
    df = pd.read_csv(path, dtype={"sentence_id":"string"})
    require_columns(df, SCORE_COLUMNS, path)
    
    # Remove rows with missing info
    df = df.dropna(subset=SCORE_COLUMNS).copy()

    # Normalize data
    df["sentence_id"] = df["sentence_id"].str.strip()
    df["cand_norm"] = df["cand_norm"].apply(normalize_word)
    df["vote_score"] = pd.to_numeric(df["vote_score"])

    # Remove empty values
    df = df[df["sentence_id"].ne("") & df["cand_norm"].ne("")].copy()

    return df


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", type=Path, default=Path("_mt_candidates_tmp.csv"))
    parser.add_argument("--key", type=Path, required=True)
    parser.add_argument("--output-thresholds", type=Path, default=Path("mt_thr_metrics.csv"))
    parser.add_argument("--num-thresholds", type=int, default=5)

    return parser.parse_args()

def main():
    args = parse_args()

    # Load predictions and correct answers
    predicted = load_candidate_scores(args.input)
    gold = load_gold_terms(args.key)

    # Evaluate all possible thresholds
    rows = []
    for thr in candidate_thresholds(args.num_thresholds):
        result = calculate_threshold_metrics(predicted, gold, thr)
        rows.append(result)

    metrics = pd.DataFrame(rows)

    # Select the best threshold by f1, precision, then recall
    metrics = metrics.sort_values(
        ["f1", "precision", "recall", "threshold"],
        ascending=[False, False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    metrics["selected_for_prediction"] = 0
    if not metrics.empty:
        metrics.loc[0, "selected_for_prediction"] = 1

    args.output_thresholds.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output_thresholds, index=False)
    print(f"Wrote threshold study to {args.output_thresholds}", flush=True)

if __name__ == "__main__":
    main()
