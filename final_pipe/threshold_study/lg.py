# -*- coding: utf-8 -*-

import argparse
import math
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import regex as re
import torch
from rapidfuzz import fuzz
from transformers import AutoModelForMaskedLM, AutoTokenizer


REQUIRED_DATASET_COLUMNS = [
    "sentence",
    "gold_terms",
    "term_type",
    "source_dataset",
    "is_single_word",
    "is_multiword",
]
WORD_RE = re.compile(r"[\p{L}\p{M}]+(?:['\u2019-][\p{L}\p{M}]+)?|[\p{N}]+")
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
        description="Run likelihood-gap scoring over MT-proposed candidates."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--mt-input",
        type=Path,
        default=None,
        help="MT all-predictions file. Defaults to output_dir/../mt_outputs/mt_preds_all.csv.",
    )
    parser.add_argument("--model-name", default="xlm-roberta-base")
    parser.add_argument("--num-thresholds", type=int, default=25)
    parser.add_argument("--features-output", type=Path, default=None)
    parser.add_argument("--thresholds-output", type=Path, default=None)
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
    return re.sub(r"[\U00010000-\U0010ffff]", "", str(text))


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


def normalize_term(text):
    return deterministic_normalize(text, keep_internal_punctuation=True)


def normalize_word(text):
    return deterministic_normalize(text, keep_internal_punctuation=True).replace(" ", "")


def clean_candidate_text(text):
    return normalize_term(text)


def tokenize_words(text):
    return [match.group(0) for match in WORD_RE.finditer(str(text))]


def normalize_candidate(row):
    cand_type = str(row.get("cand_type", "")).lower()
    if cand_type == "is_single_word":
        return normalize_word(row.get("cand_norm", row.get("candidate", "")))
    return normalize_term(row.get("cand_norm", row.get("candidate", "")))


def load_normalized_dataset(path):
    df = pd.read_csv(path)
    require_columns(df, REQUIRED_DATASET_COLUMNS, path)
    df = df.dropna(subset=["sentence", "gold_terms"]).copy()
    df["sentence"] = df["sentence"].astype(str)
    df["gold_norm"] = df["gold_terms"].apply(normalize_term)
    df["is_single_word"] = df["is_single_word"].apply(to_binary)
    df["is_multiword"] = df["is_multiword"].apply(to_binary)
    return df


def sentence_id_map(dataset_df):
    return {
        sentence: index
        for index, sentence in enumerate(dataset_df["sentence"].drop_duplicates().tolist())
    }


def gold_by_sentence_and_type(dataset_df):
    ids = sentence_id_map(dataset_df)
    output = {}
    for sentence, group in dataset_df.groupby("sentence", sort=False):
        sentence_id = str(ids[sentence])
        output[(sentence_id, "is_single_word")] = {
            normalize_word(value)
            for value in group.loc[group["is_single_word"].eq(1), "gold_norm"]
            if value
        }
        output[(sentence_id, "is_multiword")] = {
            normalize_term(value)
            for value in group.loc[group["is_multiword"].eq(1), "gold_norm"]
            if value
        }
    return output


def is_gold_candidate(candidate_norm, gold_norms):
    for gold_norm in gold_norms:
        if not gold_norm:
            continue
        if candidate_norm == gold_norm or fuzz.ratio(candidate_norm, gold_norm) >= 92:
            return 1
    return 0


def load_mt_candidates(mt_input):
    mt_df = pd.read_csv(mt_input)
    if "candidate_type" in mt_df.columns and "cand_type" not in mt_df.columns:
        mt_df = mt_df.rename(columns={"candidate_type": "cand_type"})
    require_columns(
        mt_df,
        ["sentence_id", "sentence", "candidate", "cand_norm", "cand_type", "mt_pred"],
        mt_input,
    )
    mt_df = mt_df.copy()
    mt_df["cand_type"] = mt_df["cand_type"].astype(str)
    mt_df["cand_norm"] = mt_df.apply(normalize_candidate, axis=1)
    return mt_df


def tokenized_length(text, tokenizer):
    return len(tokenizer.encode(text, add_special_tokens=True))


def max_model_length(tokenizer):
    model_max = tokenizer.model_max_length
    if model_max is None or model_max > 100000:
        return 512
    return min(int(model_max), 512)


def build_windowed_words(words, start_idx, end_idx, tokenizer, max_len):
    candidate_words = words[start_idx:end_idx]
    left = start_idx
    right = end_idx

    while left > 0 or right < len(words):
        current_words = words[left:start_idx] + candidate_words + words[end_idx:right]
        current_text = " ".join(current_words)
        if tokenized_length(current_text, tokenizer) >= max_len - 2:
            break

        left_room = left > 0
        right_room = right < len(words)
        if not left_room and not right_room:
            break

        left_distance = start_idx - left
        right_distance = right - end_idx
        if left_room and (not right_room or left_distance <= right_distance):
            left -= 1
        elif right_room:
            right += 1

    window_words = words[left:right]
    window_start_idx = start_idx - left
    window_end_idx = end_idx - left

    while tokenized_length(" ".join(window_words), tokenizer) >= max_len - 2:
        if window_start_idx > 0:
            window_words = window_words[1:]
            window_start_idx -= 1
            window_end_idx -= 1
        elif window_end_idx < len(window_words):
            window_words = window_words[:-1]
        else:
            break

    return window_words, window_start_idx, window_end_idx, left, right


def score_span_in_context(words, start_idx, end_idx, tokenizer, model):
    original_word_count = len(words)
    candidate_words = words[start_idx:end_idx]
    candidate = " ".join(candidate_words)
    candidate_norm = normalize_term(candidate)
    cand_ids = tokenizer.encode(candidate, add_special_tokens=False)

    if len(cand_ids) == 0 or not candidate_norm:
        return None

    original_start_idx = start_idx
    original_end_idx = end_idx
    max_len = max_model_length(tokenizer)
    if tokenized_length(" ".join(words), tokenizer) >= max_len - 2:
        words, start_idx, end_idx, window_start, window_end = build_windowed_words(
            words,
            start_idx,
            end_idx,
            tokenizer,
            max_len,
        )
    else:
        window_start = 0
        window_end = len(words)

    num_subwords = len(cand_ids)
    mask_span = " ".join([tokenizer.mask_token] * num_subwords)
    masked_sentence = " ".join(words[:start_idx] + [mask_span] + words[end_idx:])

    if tokenized_length(masked_sentence, tokenizer) > max_len:
        return None

    inputs = tokenizer(masked_sentence, return_tensors="pt", truncation=False)
    device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}
    mask_positions = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]

    if len(mask_positions) != num_subwords:
        return None

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits[0]
    candidate_log_prob = 0.0
    top_log_prob = 0.0
    top_tokens = []

    for mask_idx, subword_id in zip(mask_positions, cand_ids):
        log_probs = torch.log_softmax(logits[mask_idx], dim=-1)
        candidate_log_prob += log_probs[subword_id].item()
        top_id = torch.argmax(log_probs).item()
        top_log_prob += log_probs[top_id].item()
        top_tokens.append(tokenizer.decode([top_id]).strip())

    candidate_surprise = -candidate_log_prob
    gap = top_log_prob - candidate_log_prob

    return {
        "masked sentence": masked_sentence,
        "subwords": " ".join(tokenizer.convert_ids_to_tokens(cand_ids)),
        "number of subwords": num_subwords,
        "candidate probability": math.exp(candidate_log_prob),
        "candidate surprise": candidate_surprise,
        "candidate avg surprise": candidate_surprise / num_subwords,
        "top token": " ".join(top_tokens),
        "top token probability": math.exp(top_log_prob),
        "gap": gap,
        "average gap": gap / num_subwords,
        "start word index": original_start_idx,
        "end word index": original_end_idx - 1,
        "window start word index": window_start,
        "window end word index": window_end - 1,
        "windowed_context": int(window_start != 0 or window_end != original_word_count),
        "number of words": max(1, original_end_idx - original_start_idx),
    }


def candidate_span(row):
    try:
        start_idx = int(float(row.get("token_pos", "")))
    except (TypeError, ValueError):
        return None
    if "end_token_pos" in row and pd.notna(row["end_token_pos"]):
        try:
            end_idx = int(float(row["end_token_pos"])) + 1
            return start_idx, max(start_idx + 1, end_idx)
        except (TypeError, ValueError):
            pass
    token_count = len(tokenize_words(row.get("candidate", "")))
    if token_count < 1:
        token_count = len(tokenize_words(row.get("cand_norm", "")))
    return start_idx, start_idx + max(1, token_count)


def score_mt_candidates(mt_df, tokenizer, model, gold_lookup):
    rows = []
    grouped_words = {}

    for row_idx, row in mt_df.iterrows():
        if row_idx % 100 == 0:
            print(f"Scoring candidate {row_idx + 1}/{len(mt_df)}", flush=True)

        output_row = row.to_dict()
        output_row["is_gold"] = is_gold_candidate(
            row["cand_norm"],
            gold_lookup.get((str(row["sentence_id"]), str(row["cand_type"]).lower()), set()),
        )
        output_row["lg_pred"] = 0

        if to_binary(row["mt_pred"]) != 1:
            output_row["lg_score"] = 0.0
            rows.append(output_row)
            continue

        sentence_key = str(row["sentence_id"])
        if sentence_key not in grouped_words:
            grouped_words[sentence_key] = clean_candidate_text(row["sentence"]).split()
        words = grouped_words[sentence_key]
        span = candidate_span(row)
        if span is None:
            output_row["lg_score"] = np.nan
            rows.append(output_row)
            continue
        start_idx, end_idx = span
        if start_idx < 0 or end_idx > len(words) or start_idx >= end_idx:
            output_row["lg_score"] = np.nan
            rows.append(output_row)
            continue

        score = score_span_in_context(words, start_idx, end_idx, tokenizer, model)
        if score is None:
            output_row["lg_score"] = np.nan
            rows.append(output_row)
            continue
        output_row.update(score)
        output_row["lg_score"] = score["average gap"]
        rows.append(output_row)

    return pd.DataFrame(rows)


def build_thresholds(scores, num_thresholds=25):
    scores = np.asarray(scores, dtype=float)
    scores = scores[np.isfinite(scores)]
    if len(scores) == 0:
        return np.array([])
    unique_scores = np.unique(scores)
    if len(unique_scores) <= num_thresholds:
        return np.round(unique_scores, 6)
    return np.round(np.unique(np.quantile(scores, np.linspace(0, 1, num_thresholds))), 6)


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


def evaluate_thresholds(cand_df, num_thresholds):
    eval_df = cand_df[cand_df["lg_score"].notna()].copy()
    y_true = eval_df["is_gold"].astype(int).values
    scores = eval_df["lg_score"].astype(float).values
    mt_mask = eval_df["mt_pred"].apply(to_binary).eq(1).values
    rows = []

    for threshold in build_thresholds(scores, num_thresholds):
        y_pred = (mt_mask & (scores >= threshold)).astype(int)
        tp, fp, fn, tn = calculate_counts(y_true, y_pred)
        precision, recall, f1 = calculate_metrics(tp, fp, fn)
        rows.append(
            {
                "score_category": "avg gap",
                "score_column": "average gap",
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
                "selected_for_prediction": 0,
            }
        )

    threshold_df = pd.DataFrame(rows)
    if not threshold_df.empty:
        best_idx = threshold_df.sort_values(
            ["f1", "precision", "recall", "threshold"],
            ascending=[False, False, False, False],
            kind="mergesort"
        ).index[0]
        threshold_df.loc[best_idx, ["is_best", "selected_for_prediction"]] = 1
    return threshold_df


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mt_input = args.mt_input or args.output_dir.parent / "mt_outputs" / "mt_preds_all.csv"
    features_output = args.features_output or args.output_dir / "lg_scoring_breakdown.csv"
    thresholds_output = args.thresholds_output or args.output_dir / "lg_thr_metrics.csv"

    print("Starting likelihood-gap run", flush=True)
    print(f"Dataset: {args.dataset}", flush=True)
    print(f"MT input: {mt_input}", flush=True)
    print(f"Output directory: {args.output_dir}", flush=True)
    print(f"Model name: {args.model_name}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, flush=True)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0), flush=True)
    model = AutoModelForMaskedLM.from_pretrained(args.model_name).to(device)
    model.eval()

    dataset_df = load_normalized_dataset(args.dataset)
    mt_df = load_mt_candidates(mt_input)
    candidates_df = score_mt_candidates(
        mt_df,
        tokenizer,
        model,
        gold_by_sentence_and_type(dataset_df),
    )
    candidates_df.to_csv(features_output, index=False)
    print(f"Wrote {len(candidates_df)} candidate scores to {features_output}", flush=True)

    threshold_df = evaluate_thresholds(candidates_df, args.num_thresholds)
    threshold_df.to_csv(thresholds_output, index=False)
    print(f"Wrote threshold metrics to {thresholds_output}", flush=True)


if __name__ == "__main__":
    main()
