"""Build multiword candidates from consecutive single word predictions.
For a spang: A B C, possible multiword candidates are AB, BC, ABC. 

Repeated occurrences remain separate output rows. Metrics follow the current
pipeline convention and match unique terms by sentence_id plus normalized text.
"""

import argparse
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

KEY_COLUMNS = [
    "sentence_id",
    "special_term",
    "term_type"
]

SW_COLUMNS = [
    "sentence_id",
    "sentence",
    "candidate",
    "token_pos",
    "vote_score",
    "mt_pred"
]

MW_COLUMNS = [
    "sentence_id",
    "sentence",
    "candidate",
    "cand_norm",
    "cand_type",
    "token_pos",
    "end_token_pos",
    "start_char",
    "end_char",
    "vote_score",
    "mt_pred"
]

# Calculate metrics for candidate type
def metrics_by_type(predictions, gold, cand_type):
    universe = sorted(predictions | gold, key=str)
    y_true = [int(identity in gold) for identity in universe]
    y_pred = [int(identity in predictions) for identity in universe]

    if universe:
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    else:
        precision = recall = f1 = 0.0

    return {
    "candidate_type": cand_type,
    "true_positive": len(predictions & gold),
    "false_positive": len(predictions - gold),
    "false_negative": len(gold - predictions),
    "predicted_count": len(predictions),
    "gold_count": len(gold),
    "precision": float(precision),
    "recall": float(recall),
    "f1": float(f1),
}


# Calculate metrics for both candidate types
def calculate_metrics(words, phrases, gold):
    # Save single word predictions
    positive_words = words[words["mt_pred"].eq(1)]
    predicted_single = {
        (row.sentence_id, normalize_term(row.candidate))
        for row in positive_words.itertuples(index=False)
    }

    # Save multiword predictions
    predicted_multi = {
        (row.sentence_id, row.cand_norm)
        for row in phrases.itertuples(index=False)
        if int(row.mt_pred) == 1
    }

    # Key for single and multiword terms
    gold_single = gold["single_word"]
    gold_multi = gold["multiword"]

    # Calculate metrics by candidate type
    return pd.DataFrame(
        [
            metrics_by_type(predicted_single, gold_single, "single_word"),
            metrics_by_type(predicted_multi, gold_multi, "multiword"),
            metrics_by_type(predicted_single | predicted_multi, gold_single | gold_multi, "both")
        ]
    )


# Detect the longest possible conescutive spans of SW predictions
def positive_runs(group):
    run = []
    previous_position = None

    for row in group.sort_values("token_pos", kind="mergesort").itertuples(index=False):
        position = int(row.token_pos)
        is_consecutive = (
            previous_position is not None
            and position == previous_position + 1
        )

        if int(row.mt_pred) == 1:
            if run and not is_consecutive:
                if len(run) >= 2:
                    yield run
                run = []
            run.append(row)
        else:
            if len(run) >= 2:
                yield run
            run = []

        previous_position = position

    if len(run) >= 2:
        yield run


# For a single run of positive SW predictions, create every possible subspan of 2 or more words
def build_phrase_rows(words):
    rows = []

    # For every unique sentence, save its multiword candidates
    for (sentence_id, sentence), group in words.groupby(["sentence_id", "sentence"], sort=False):
        sentence = str(sentence)

        # Detect a positive run of consecutive SWs
        for run in positive_runs(group):
            # A run A B C produces A B, B C, and A B C.
            # Select the phrase's first word
            for start_index in range(len(run) - 1):
                # Select a possible ending word after the start
                for end_index in range(start_index + 1, len(run)):
                    pieces = run[start_index : end_index + 1]
                    start_char = int(pieces[0].start_char)
                    end_char = int(pieces[-1].end_char)

                    if start_char < 0 or end_char > len(sentence):
                        raise ValueError(f"Invalid character offsets for sentence {sentence_id}.")

                    candidate = sentence[start_char:end_char]
                    rows.append(
                        {
                            "sentence_id": sentence_id,
                            "sentence": sentence,
                            "candidate": candidate,
                            "cand_norm": normalize_term(candidate),
                            "cand_type": "multiword",
                            "token_pos": int(pieces[0].token_pos),
                            "end_token_pos": int(pieces[-1].token_pos),
                            "start_char": start_char,
                            "end_char": end_char,
                            "vote_score": float(np.mean([float(piece.vote_score) for piece in pieces])), # Represente MW disagreement score as the average of its SW
                            "mt_pred": 1,
                        }
                    )

    return pd.DataFrame(rows, columns=MW_COLUMNS)


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


# Helper function to normalize strings
def normalize_term(value):
    text = unicodedata.normalize("NFKC", str(value))
    text = re.sub(r"\s+", " ", text)
    return text.strip().casefold()


# Helper function to check if required columns exist in input
def require_columns(df, reqs, path):
    missing = []
    for col in reqs:
        if col not in df.columns:
            missing.append(col)
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")


# Load single and multiword gold terms from the key
def load_gold_terms(path):
    df = pd.read_csv(path, dtype={"sentence_id": "string"})
    require_columns(df, KEY_COLUMNS, path)

    # Remove rows with missing info
    df = df.dropna(subset=KEY_COLUMNS).copy()

    # Normalize entries
    df["sentence_id"] = df["sentence_id"].str.strip()
    df["special_term_norm"] = df["special_term"].apply(normalize_term)
    df["term_type"] = df["term_type"].apply(normalize_type)

    # Remove empty values
    df = df[df["sentence_id"].ne("") & df["special_term_norm"].ne("")]
    
    # Load gold terms by candidate type
    single_terms = set()
    multiword_terms = set()

    for row in df.itertuples(index=False):
        identity = (row.sentence_id, row.special_term_norm)

        if row.term_type == "single_word": single_terms.add(identity)
        elif row.term_type == "multiword": multiword_terms.add(identity)

    gold_terms = {
        "single_word": single_terms,
        "multiword": multiword_terms
    }
    return gold_terms


# Load single word predictions
def load_word_predictions(path):
    df = pd.read_csv(path, dtype={"sentence_id": "string"})
    require_columns(df, SW_COLUMNS, path)

    # Remove rows with missing info
    df = df.dropna(subset=SW_COLUMNS + ["start_char", "end_char"]).copy()
    
    # Normalize entries
    df["sentence_id"] = df["sentence_id"].str.strip()
    df["sentence"] = df["sentence"].astype(str)
    for column in ["token_pos", "start_char", "end_char", "mt_pred"]:
        df[column] = pd.to_numeric(df[column], errors="raise").astype(int)
    df["vote_score"] = pd.to_numeric(df["vote_score"], errors="raise")

    return df


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--word-preds", type=Path, default=Path("mt_preds_sw.csv"))
    parser.add_argument("--key", type=Path, required=True)
    parser.add_argument("--phrases", type=Path, default=Path("mt_preds_mw.csv"))
    parser.add_argument("--combined", type=Path, default=Path("mt_preds_all.csv"))
    parser.add_argument("--metrics", type=Path, default=Path("mt_metrics.csv"))
    
    return parser.parse_args()


def main():
    args = parse_args()

    # Save key
    gold = load_gold_terms(args.key)

    # Build multi word predictions
    words = load_word_predictions(args.word_preds)    
    phrases = build_phrase_rows(words)
    args.phrases.parent.mkdir(parents=True, exist_ok=True)
    phrases.to_csv(args.phrases, index=False)
    print(f"Wrote {len(phrases)} multiword occurrences to {args.phrases}", flush=True)

    # Save full set of SW and MW predictions from Model Agreement
    combined = pd.concat([words, phrases], ignore_index=True, sort=False)
    args.combined.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.combined, index=False)
    print(f"Wrote {len(combined)} combined occurrences to {args.combined}", flush=True)

    # Save metrics for both candidate types
    metrics = calculate_metrics(words, phrases, gold)
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.metrics, index=False)
    print(metrics.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
