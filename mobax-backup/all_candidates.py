# -*- coding: utf-8 -*-
"""Build the complete candidate inventory used by standalone SC/LG/MA runs.

This deliberately emits ``mt_pred = 1`` for every candidate.  The existing
threshold-study implementations can therefore retain their original scoring
logic while evaluating every word and every consecutive multiword candidate,
instead of only candidates accepted by MT.
"""

import argparse
from pathlib import Path

import pandas as pd

import mt_consecutive as mtc


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate all word and consecutive phrase candidates."
    )
    parser.add_argument("--dataset", type=Path, default=Path("genz_normalized.csv"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("all_candidates.csv"),
    )
    return parser.parse_args()


def _load_dataset(dataset_path, limit_sentences=None):
    dataset = pd.read_csv(dataset_path)
    mtc.require_columns(dataset, mtc.REQUIRED_DATASET_COLUMNS, dataset_path)
    dataset = dataset.dropna(subset=["sentence", "gold_terms"]).copy()
    dataset["sentence"] = dataset["sentence"].map(mtc.clean_sentence)

    if limit_sentences is not None:
        retained = dataset["sentence"].drop_duplicates().head(limit_sentences)
        dataset = dataset[dataset["sentence"].isin(set(retained))].copy()

    sentence_ids = {
        sentence: index
        for index, sentence in enumerate(dataset["sentence"].drop_duplicates())
    }
    dataset["sentence_id"] = dataset["sentence"].map(sentence_ids)
    dataset["_sentence_id_str"] = dataset["sentence_id"].astype(str)
    return dataset


def _single_word_gold(dataset):
    gold = {}
    for _, row in dataset.iterrows():
        if not mtc.to_binary(row["is_single_word"]):
            continue
        key = row["_sentence_id_str"]
        value = mtc.normalize_word(row["gold_terms"])
        if value:
            gold.setdefault(key, set()).add(value)
    return gold


def _build_word_candidates(dataset):
    gold_by_sentence = _single_word_gold(dataset)
    rows = []

    sentences = dataset[["sentence_id", "_sentence_id_str", "sentence"]].drop_duplicates(
        subset=["sentence_id"]
    )
    for _, sentence_row in sentences.iterrows():
        sentence_id = sentence_row["sentence_id"]
        sentence_id_str = sentence_row["_sentence_id_str"]
        sentence = sentence_row["sentence"]
        tokens = mtc.tokenize_words(sentence)
        gold_terms = gold_by_sentence.get(sentence_id_str, set())

        for token_pos, token in enumerate(tokens):
            cand_norm = mtc.normalize_word(token)
            if not cand_norm:
                continue
            rows.append(
                {
                    "sentence_id": sentence_id,
                    "sentence": sentence,
                    "candidate": token,
                    "cand_norm": cand_norm,
                    "cand_type": "is_single_word",
                    "is_gold": int(cand_norm in gold_terms),
                    "token_pos": token_pos,
                    "end_token_pos": token_pos,
                    "component_token_positions": str(token_pos),
                    "mt_score": 1.0,
                    "mt_pred": 1,
                }
            )

    return pd.DataFrame(rows)


def _multiword_gold(dataset):
    gold = {}
    for _, row in dataset.iterrows():
        if not mtc.to_binary(row["is_multiword"]):
            continue
        key = row["_sentence_id_str"]
        value = mtc.normalize_text(row["gold_terms"])
        if value:
            gold.setdefault(key, set()).add(value)
    return gold


def build_all_candidates(dataset_path, limit_sentences=None):
    """Return all word candidates plus all consecutive phrase candidates."""
    dataset = _load_dataset(dataset_path, limit_sentences=limit_sentences)
    word_df = _build_word_candidates(dataset)

    if word_df.empty:
        return pd.DataFrame(columns=mtc.output_columns(pd.DataFrame()))

    word_df["_sentence_id_str"] = word_df["sentence_id"].astype(str)
    phrase_df = mtc.build_phrase_rows(word_df, _multiword_gold(dataset))

    if not phrase_df.empty:
        phrase_df["end_token_pos"] = (
            phrase_df["token_pos"].astype(int)
            + phrase_df["candidate"].map(lambda value: len(mtc.tokenize_words(value)))
            - 1
        )
        phrase_df["component_token_positions"] = phrase_df.apply(
            lambda row: ",".join(
                str(position)
                for position in range(
                    int(row["token_pos"]), int(row["end_token_pos"]) + 1
                )
            ),
            axis=1,
        )

    combined = pd.concat(
        [word_df.drop(columns=["_sentence_id_str"]), phrase_df],
        ignore_index=True,
        sort=False,
    )
    preferred = [
        "sentence_id",
        "sentence",
        "candidate",
        "cand_norm",
        "cand_type",
        "is_gold",
        "token_pos",
        "end_token_pos",
        "component_token_positions",
        "mt_score",
        "mt_pred",
    ]
    columns = [column for column in preferred if column in combined.columns]
    columns.extend(column for column in combined.columns if column not in columns)
    return combined[columns]


def main():
    args = parse_args()
    candidates = build_all_candidates(args.dataset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(args.output, index=False)
    print(f"Wrote {len(candidates)} all-candidate rows to {args.output}", flush=True)


if __name__ == "__main__":
    main()
