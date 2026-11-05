# -*- coding: utf-8 -*-

"""Compare LLM prompt predictions with a hidden specialized-language key.
Predictions are evaluated by sentence_id and special_term only."""

import argparse
import re
import unicodedata
from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score


PREDICTION_COLUMNS = [
    "sentence_id",
    "sentence",
    "special_term",
    "term_type"
]
KEY_COLUMNS = PREDICTION_COLUMNS

# Calculate metrics over the union of gold and predictions.
def metrics_for_types(predicted, gold, included_types, label):
    
    # Select candidate type for metric evaluation
    predicted_subset = {
        identity for identity, term_type in predicted.items() 
        if term_type in included_types
    }
    gold_subset = {
        identity for identity, term_type in gold.items() 
        if term_type in included_types
    }
    universe = sorted(predicted_subset | gold_subset, key=str)

    y_true = [int(identity in gold_subset) for identity in universe]
    y_pred = [int(identity in predicted_subset) for identity in universe]
    
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    true_positive = len(predicted_subset & gold_subset)
    false_positive = len(predicted_subset - gold_subset)
    false_negative = len(gold_subset - predicted_subset)

    return {
        "candidate_type": label,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "predicted_count": len(predicted_subset),
        "gold_count": len(gold_subset),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

# Helper function to ensure a shared common name for candidate types
def normalize_type(value):
    cleaned = str(value).strip().casefold().replace("-", "_").replace(" ", "_")
    aliases = {
        "single": "single_word",
        "word": "single_word",
        "singleword": "single_word",
        "single_word": "single_word",
        "multi": "multiword",
        "phrase": "multiword",
        "multi_word": "multiword",
        "multiword": "multiword",
    }
    if cleaned not in aliases:
        raise ValueError(f"Unknown term_type value: {value!r}")
    return aliases[cleaned]

# Helper function to normalize a term for duplicate detection and evaluation
def normalize_term(term):
    # Make capitalization and white space irrelevant
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


# Load unique labeled spans; duplicate CSV rows count only once.
def load_span_set(path, required_columns):
    df = pd.read_csv(path, dtype={"sentence_id":"string"})
    require_columns(df, required_columns, path)

    # Remove rows with missing info
    df = df.dropna(
        subset=[
            "sentence_id",
            "special_term",
            "term_type"
        ]
    ).copy()

    df["sentence_id"] = df["sentence_id"].str.strip()
    df["special_term_norm"] = df["special_term"].apply(normalize_term)    
    df["term_type"] = df["term_type"].apply(normalize_type) # One label only
    
    # Remove empty values
    df = df[df["sentence_id"].ne("") & df["special_term_norm"].ne("")].copy()

    term_map = {}

    for row in df.itertuples(index=False):
        identity = (row.sentence_id, row.special_term_norm)
        existing_type = term_map.get(identity)

        if existing_type is not None and existing_type != row.term_type:
            raise ValueError(f"Too many term types assigned")
        
        term_map[identity] = row.term_type

    return term_map


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--key", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()



def main():
    args = parse_args()
    predicted = load_span_set(args.predictions, PREDICTION_COLUMNS)
    gold = load_span_set(args.key, KEY_COLUMNS)

    # Run metrics by candidate type
    metrics = pd.DataFrame(
        [
            metrics_for_types(
                predicted, gold, {"single_word"}, "single_word"
            ),
            metrics_for_types(
                predicted, gold, {"multiword"}, "multiword"
            ),
            metrics_for_types(
                predicted,
                gold,
                {"single_word", "multiword"},
                "both",
            ),
        ]
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output, index=False)
    print(metrics.to_string(index=False), flush=True)
    print(f"Wrote metrics to {args.output}", flush=True)


if __name__ == "__main__":
    main()
