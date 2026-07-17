# -*- coding: utf-8 -*-

import argparse
import unicodedata
from pathlib import Path

import pandas as pd
import regex as re
from rapidfuzz import fuzz

METHODS = {"sc": ("sc_score", "sc_pred"), "lg": ("lg_score", "lg_pred"), "ma": ("ma_score", "ma_pred")}
REQUIRED_DATASET_COLUMNS = ["sentence", "gold_terms", "term_type", "source_dataset", "is_single_word", "is_multiword"]
APOSTROPHES = str.maketrans({"\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201b": "'", "\u2032": "'", "\u2035": "'", "`": "'", "\u00b4": "'"})
HYPHENS = str.maketrans({"\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-", "\u2014": "-", "\u2015": "-", "\u2212": "-"})


def parse_args():
    parser = argparse.ArgumentParser(description="Compile method scores and predictions for late fusion.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--sc-input", type=Path, required=True)
    parser.add_argument("--lg-input", type=Path, required=True)
    parser.add_argument("--ma-input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
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


def deterministic_normalize(text, keep_internal_punctuation=True):
    text = unicodedata.normalize("NFKC", str(text))
    text = text.translate(APOSTROPHES).translate(HYPHENS).casefold().strip()
    text = re.sub(r"\s+", " ", text)
    if keep_internal_punctuation:
        text = re.sub(r"[^\p{L}\p{N}'\-\s]+", " ", text)
    else:
        text = re.sub(r"[^\p{L}\p{N}\s]+", " ", text)
    text = re.sub(r"^[^\p{L}\p{N}]+|[^\p{L}\p{N}]+$", "", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_text(text):
    return deterministic_normalize(text, keep_internal_punctuation=True)


def normalize_word(text):
    return deterministic_normalize(text, keep_internal_punctuation=True).replace(" ", "")


def normalize_candidate(row):
    if str(row.get("cand_type", "")).lower() == "is_single_word":
        return normalize_word(row.get("cand_norm", row.get("candidate", "")))
    return normalize_text(row.get("cand_norm", row.get("candidate", "")))


def load_gold_lookup(dataset_path):
    df = pd.read_csv(dataset_path)
    require_columns(df, REQUIRED_DATASET_COLUMNS, dataset_path)
    df = df.dropna(subset=["sentence", "gold_terms"]).copy()
    df["sentence"] = df["sentence"].astype(str)
    df["gold_norm"] = df["gold_terms"].apply(normalize_text)
    df["is_single_word"] = df["is_single_word"].apply(to_binary)
    df["is_multiword"] = df["is_multiword"].apply(to_binary)
    sentence_ids = {sentence: index for index, sentence in enumerate(df["sentence"].drop_duplicates().tolist())}
    lookup = {}
    for sentence, group in df.groupby("sentence", sort=False):
        sid = str(sentence_ids[sentence])
        lookup[(sid, "is_single_word")] = {normalize_word(value) for value in group.loc[group["is_single_word"].eq(1), "gold_norm"] if value}
        lookup[(sid, "is_multiword")] = {normalize_text(value) for value in group.loc[group["is_multiword"].eq(1), "gold_norm"] if value}
    return lookup


def is_gold_candidate(candidate_norm, gold_norms):
    for gold_norm in gold_norms:
        if candidate_norm == gold_norm or fuzz.ratio(candidate_norm, gold_norm) >= 92:
            return 1
    return 0


def load_method(path, method):
    score_col, pred_col = METHODS[method]
    df = pd.read_csv(path)
    if "candidate_type" in df.columns and "cand_type" not in df.columns:
        df = df.rename(columns={"candidate_type": "cand_type"})
    require_columns(df, ["sentence_id", "sentence", "candidate", "cand_norm", "cand_type", "is_gold", score_col, pred_col], path)
    keep = ["sentence_id", "sentence", "candidate", "cand_norm", "cand_type", "is_gold", "token_pos", "end_token_pos", "component_token_positions", score_col, pred_col]
    df = df[[column for column in keep if column in df.columns]].copy()
    df["sentence_id"] = df["sentence_id"].astype(str)
    df["cand_type"] = df["cand_type"].astype(str)
    df["cand_norm"] = df.apply(normalize_candidate, axis=1)
    token_pos = pd.to_numeric(df.get("token_pos", pd.Series(index=df.index)), errors="coerce").fillna(-1).astype(int).astype(str)
    df["_key"] = df["sentence_id"] + "||" + df["cand_type"].str.lower() + "||" + df["cand_norm"].astype(str) + "||" + token_pos
    return df


def main():
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    gold_lookup = load_gold_lookup(args.dataset)
    frames = []
    for method, path in [("sc", args.sc_input), ("lg", args.lg_input), ("ma", args.ma_input)]:
        df = load_method(path, method)
        score_col, pred_col = METHODS[method]
        base_cols = ["_key", "sentence_id", "sentence", "candidate", "cand_norm", "cand_type", "token_pos", "end_token_pos", "component_token_positions", score_col, pred_col]
        frames.append(df[[column for column in base_cols if column in df.columns]])

    compiled = frames[0]
    for frame in frames[1:]:
        compiled = compiled.merge(frame, on="_key", how="outer", suffixes=("", "_new"))
        for column in ["sentence_id", "sentence", "candidate", "cand_norm", "cand_type", "token_pos", "end_token_pos", "component_token_positions"]:
            new_col = f"{column}_new"
            if new_col in compiled.columns:
                compiled[column] = compiled[column].fillna(compiled[new_col])
                compiled = compiled.drop(columns=[new_col])

    for _method, (score_col, pred_col) in METHODS.items():
        compiled[score_col] = pd.to_numeric(compiled.get(score_col), errors="coerce").fillna(0.0)
        compiled[pred_col] = compiled.get(pred_col, 0).apply(to_binary)

    compiled["is_gold"] = compiled.apply(lambda row: is_gold_candidate(row["cand_norm"], gold_lookup.get((str(row["sentence_id"]), str(row["cand_type"]).lower()), set())), axis=1)
    compiled = compiled.drop(columns=["_key"])
    compiled.to_csv(args.output, index=False)
    print(f"Wrote compiled predictions to {args.output}", flush=True)


if __name__ == "__main__":
    main()
