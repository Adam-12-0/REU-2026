# -*- coding: utf-8 -*-

import argparse
import unicodedata
from pathlib import Path

import pandas as pd
import regex as re


REQUIRED_DATASET_COLUMNS = [
    "sentence",
    "gold_terms",
    "term_type",
    "source_dataset",
    "is_single_word",
    "is_multiword",
]

WORD_RE = re.compile(r"[\p{L}\p{M}]+(?:['\u2019-][\p{L}\p{M}]+)?|[\p{N}]+")
EMOJI_RE = re.compile(r"\p{Emoji_Presentation}|\p{Extended_Pictographic}")
MOJIBAKE_EMOJI_RE = re.compile(r"ÃƒÂ°Ã…Â¸\S*")
APOSTROPHES = str.maketrans({
    "\u2018": "'",
    "\u2019": "'",
    "\u201a": "'",
    "\u201b": "'",
    "\u2032": "'",
    "\u2035": "'",
    "`": "'",
    "\u00b4": "'",
})
HYPHENS = str.maketrans({
    "\u2010": "-",
    "\u2011": "-",
    "\u2012": "-",
    "\u2013": "-",
    "\u2014": "-",
    "\u2015": "-",
    "\u2212": "-",
})


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build MT multiword predictions from consecutive specialized word predictions."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--phrase-output", type=Path, required=True)
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


def remove_emojis(text):
    text = EMOJI_RE.sub("", str(text))
    return MOJIBAKE_EMOJI_RE.sub("", text)


def deterministic_normalize(text, keep_internal_punctuation=True):
    text = remove_emojis(text)
    text = unicodedata.normalize("NFKC", str(text))
    text = text.translate(APOSTROPHES).translate(HYPHENS)
    text = text.casefold().strip()
    text = re.sub(r"\s+", " ", text)
    if keep_internal_punctuation:
        text = re.sub(r"[^\p{L}\p{N}'\-\s]+", " ", text)
    else:
        text = re.sub(r"[^\p{L}\p{N}\s]+", " ", text)
    text = re.sub(r"^[^\p{L}\p{N}]+|[^\p{L}\p{N}]+$", "", text)
    return re.sub(r"\s+", " ", text).strip()


def clean_sentence(text):
    return re.sub(r"\s+", " ", remove_emojis(text)).strip()


def normalize_text(text):
    return deterministic_normalize(text, keep_internal_punctuation=True)


def normalize_word(text):
    return deterministic_normalize(text, keep_internal_punctuation=True).replace(" ", "")


def tokenize_words(text):
    text = remove_emojis(text)
    return [match.group(0) for match in WORD_RE.finditer(str(text))]


def load_multiword_gold(dataset_path):
    df = pd.read_csv(dataset_path)
    require_columns(df, REQUIRED_DATASET_COLUMNS, dataset_path)
    df = df.dropna(subset=["sentence", "gold_terms"]).copy()
    df["sentence"] = df["sentence"].astype(str)
    df["is_multiword"] = df["is_multiword"].apply(to_binary)
    df["gold_norm"] = df["gold_terms"].apply(normalize_text)

    sentence_ids = {
        sentence: index
        for index, sentence in enumerate(df["sentence"].drop_duplicates().tolist())
    }
    gold_by_sentence = {}
    for sentence, group in df[df["is_multiword"].eq(1)].groupby("sentence", sort=False):
        sentence_id = str(sentence_ids[sentence])
        gold_by_sentence[sentence_id] = {
            value
            for value in group["gold_norm"]
            if value and len(tokenize_words(value)) >= 2
        }
    return gold_by_sentence


def is_gold_phrase(phrase_norm, gold_norms):
    return int(normalize_text(phrase_norm) in {normalize_text(gold) for gold in gold_norms})


def assign_token_positions(group):
    sentence = remove_emojis(group["sentence"].iloc[0])
    token_matches = list(WORD_RE.finditer(str(sentence)))
    sentence_tokens = [normalize_word(match.group(0)) for match in token_matches]
    next_search_at = 0
    positions = []
    token_starts = []
    token_ends = []

    for _, row in group.iterrows():
        candidate_norm = normalize_word(row.get("cand_norm", ""))
        if not candidate_norm:
            candidate_norm = normalize_word(row["candidate"])

        position = None
        token_start = None
        token_end = None
        for index in range(next_search_at, len(sentence_tokens)):
            if sentence_tokens[index] == candidate_norm:
                position = index
                token_start = token_matches[index].start()
                token_end = token_matches[index].end()
                next_search_at = index + 1
                break
        positions.append(position)
        token_starts.append(token_start)
        token_ends.append(token_end)

    positioned = group.copy()
    positioned["_token_pos"] = positions
    positioned["_token_start"] = token_starts
    positioned["_token_end"] = token_ends
    positioned["_span_sentence"] = sentence
    positioned["_fallback_pos"] = range(len(positioned))
    return positioned


def phrase_from_rows(rows):
    return clean_sentence(" ".join(str(row["candidate"]).strip() for row in rows if str(row["candidate"]).strip()))


def phrase_norm_from_rows(rows):
    return normalize_text(" ".join(str(row["cand_norm"]).strip() for row in rows))


def consecutive_runs(group):
    pred_rows = []
    for _, row in group.iterrows():
        if to_binary(row["mt_pred"]) != 1:
            if pred_rows:
                yield pred_rows
                pred_rows = []
            continue

        if not pred_rows:
            pred_rows = [row]
            continue

        prev_pos = pred_rows[-1]["_token_pos"]
        curr_pos = row["_token_pos"]
        adjacent = pd.notna(prev_pos) and pd.notna(curr_pos) and int(curr_pos) == int(prev_pos) + 1
        prev_end = pred_rows[-1]["_token_end"]
        curr_start = row["_token_start"]
        if adjacent and pd.notna(prev_end) and pd.notna(curr_start):
            between_tokens = str(row["_span_sentence"])[int(prev_end):int(curr_start)]
            adjacent = "," not in between_tokens

        if adjacent:
            pred_rows.append(row)
        else:
            if pred_rows:
                yield pred_rows
            pred_rows = [row]

    if pred_rows:
        yield pred_rows


def build_phrase_rows(word_df, gold_by_sentence):
    output_rows = []
    for sentence_id, group in word_df.groupby("_sentence_id_str", sort=False):
        group = assign_token_positions(group)
        group = group.sort_values(
            by=["_token_pos", "_fallback_pos"],
            na_position="last",
            kind="mergesort",
        )
        gold_norms = gold_by_sentence.get(sentence_id, set())
        seen_sentence_phrases = set()

        for run in consecutive_runs(group):
            if len(run) < 2:
                continue
            for start in range(len(run)):
                for end in range(start + 2, len(run) + 1):
                    phrase_rows = run[start:end]
                    phrase = phrase_from_rows(phrase_rows)
                    phrase_norm = phrase_norm_from_rows(phrase_rows)
                    if not phrase_norm or phrase_norm in seen_sentence_phrases:
                        continue
                    seen_sentence_phrases.add(phrase_norm)
                    first_row = phrase_rows[0]
                    output_rows.append(
                        {
                            "sentence_id": first_row["sentence_id"],
                            "sentence": clean_sentence(first_row["sentence"]),
                            "candidate": phrase,
                            "cand_norm": phrase_norm,
                            "cand_type": "is_multiword",
                            "is_gold": is_gold_phrase(phrase_norm, gold_norms),
                            "token_pos": first_row.get("token_pos", first_row["_token_pos"]),
                            "mt_score": 1.0,
                            "mt_pred": 1,
                        }
                    )
    return pd.DataFrame(output_rows)


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
    y_pred = df["mt_pred"].astype(int)
    tp, fp, fn = calculate_counts(y_true, y_pred)
    precision, recall, f1 = calculate_metrics(tp, fp, fn)
    return {
        "cand_type": cand_type,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def write_metrics(df, output_path):
    rows = [metric_row(cand_type, group) for cand_type, group in df.groupby("cand_type", sort=True)]
    rows.append(metric_row("all", df))
    pd.DataFrame(rows).to_csv(output_path, index=False)


def output_columns(*dfs):
    preferred = [
        "sentence_id",
        "sentence",
        "candidate",
        "cand_norm",
        "cand_type",
        "is_gold",
        "token_pos",
        "mt_score",
        "mt_pred",
    ]
    columns = [column for column in preferred if any(column in df.columns for df in dfs)]
    for df in dfs:
        for column in df.columns:
            if column not in columns and not column.startswith("_"):
                columns.append(column)
    return columns


def main():
    args = parse_args()
    word_df = pd.read_csv(args.input)
    if "candidate_type" in word_df.columns and "cand_type" not in word_df.columns:
        word_df = word_df.rename(columns={"candidate_type": "cand_type"})
    require_columns(
        word_df,
        ["sentence_id", "sentence", "candidate", "cand_norm", "cand_type", "is_gold", "mt_pred"],
        args.input,
    )

    word_df = word_df.copy()
    word_df["_sentence_id_str"] = word_df["sentence_id"].astype(str)
    word_df = word_df[word_df["cand_type"].astype(str).str.lower().eq("is_single_word")].copy()

    gold_by_sentence = load_multiword_gold(args.dataset)
    phrase_df = build_phrase_rows(word_df, gold_by_sentence)
    phrase_columns = output_columns(phrase_df)
    if phrase_df.empty:
        phrase_df = pd.DataFrame(columns=phrase_columns)
    else:
        phrase_df = phrase_df[phrase_columns]
    phrase_df.to_csv(args.phrase_output, index=False)

    word_output_df = word_df.drop(columns=[column for column in word_df.columns if column.startswith("_")])
    columns = output_columns(word_output_df, phrase_df)
    combined_df = pd.concat(
        [
            word_output_df.reindex(columns=columns),
            phrase_df.reindex(columns=columns),
        ],
        ignore_index=True,
    )
    combined_df.to_csv(args.all_output, index=False)
    write_metrics(combined_df, args.metrics_output)

    print(f"Wrote {len(phrase_df)} multiword rows to {args.phrase_output}", flush=True)
    print(f"Wrote {len(combined_df)} rows to {args.all_output}", flush=True)
    print(f"Wrote metrics to {args.metrics_output}", flush=True)


if __name__ == "__main__":
    main()
