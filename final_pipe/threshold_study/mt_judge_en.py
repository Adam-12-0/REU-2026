# -*- coding: utf-8 -*-

import argparse
import os
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import regex as re
import spacy
import torch
from rapidfuzz import fuzz
from sklearn.metrics import precision_recall_fscore_support
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline


REQUIRED_DATASET_COLUMNS = [
    "sentence",
    "gold_terms",
    "term_type",
    "source_dataset",
    "is_single_word",
    "is_multiword",
]

DEFAULT_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"

LANGUAGES = [
    ("spanish", "Spanish"),
    ("arabic", "Arabic"),
    ("chinese", "Chinese"),
    ("japanese", "Japanese"),
]

WORD_RE = re.compile(r"[\p{L}\p{M}]+(?:['\u2019-][\p{L}\p{M}]+)?|[\p{N}]+")
EMOJI_RE = re.compile(r"\p{Emoji_Presentation}|\p{Extended_Pictographic}")
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
        description="Run four-language machine-translation judge on a normalized dataset."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-id", default=DEFAULT_MODEL)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--limit-rows", type=int, default=None)
    parser.add_argument("--fuzzy-threshold", type=int, default=93)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-thresholds", type=int, default=25)
    parser.add_argument("--spacy-model", default="en_core_web_sm")
    return parser.parse_args()


def require_columns(df, columns, path):
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")


def to_binary(value):
    if pd.isna(value):
        return 0
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "t", "yes", "y"}:
        return 1
    if normalized in {"0", "false", "f", "no", "n", ""}:
        return 0
    return int(float(normalized) > 0)


def remove_emojis(text):
    return EMOJI_RE.sub("", str(text))


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
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_text(text):
    return deterministic_normalize(text, keep_internal_punctuation=True)


def normalize_word(text):
    return deterministic_normalize(text, keep_internal_punctuation=True).replace(" ", "")


def tokenize_words(text):
    text = remove_emojis(text)
    return [match.group(0) for match in WORD_RE.finditer(str(text))]


def normalized_word_set(text):
    return {normalize_word(word) for word in tokenize_words(text) if normalize_word(word)}


def split_gold_terms(value):
    if pd.isna(value) or str(value).strip() == "":
        return []
    return [
        normalize_text(term)
        for term in re.split(r";|\|", str(value))
        if normalize_text(term)
    ]


def is_gold_candidate(candidate_norm, gold_norms):
    for gold_norm in gold_norms:
        if not gold_norm:
            continue
        if candidate_norm == gold_norm or fuzz.ratio(candidate_norm, gold_norm) >= 92:
            return 1
    return 0


def load_normalized_dataset(path, limit_rows):
    df = pd.read_csv(path)
    require_columns(df, REQUIRED_DATASET_COLUMNS, path)
    df = df.dropna(subset=["sentence", "gold_terms"]).copy()
    df["sentence"] = df["sentence"].astype(str)
    df["gold_norm"] = df["gold_terms"].apply(normalize_text)
    df["is_single_word"] = df["is_single_word"].apply(to_binary)

    if limit_rows is not None:
        unique_sentences = df["sentence"].drop_duplicates().head(limit_rows)
        df = df[df["sentence"].isin(unique_sentences)].copy()

    rows = []
    sentence_ids = {
        sentence: index
        for index, sentence in enumerate(df["sentence"].drop_duplicates().tolist())
    }
    for sentence, group in df.groupby("sentence", sort=False):
        gold_word_norms = sorted(
            {
                value
                for value in group.loc[group["is_single_word"].eq(1), "gold_norm"]
                if value
            }
        )
        rows.append(
            {
                "sentence_id": sentence_ids[sentence],
                "sentence": sentence,
                "gold_word_norms": gold_word_norms,
                "gold_terms": "; ".join(gold_word_norms),
                "term_type": ";".join(sorted(group["term_type"].astype(str).unique())),
                "source_dataset": ";".join(sorted(group["source_dataset"].astype(str).unique())),
            }
        )
    return pd.DataFrame(rows)


def named_entity_word_set(sentence, nlp):
    doc = nlp(str(sentence))
    entity_words = set()
    for ent in doc.ents:
        for word in tokenize_words(ent.text):
            word_norm = normalize_word(word)
            if word_norm:
                entity_words.add(word_norm)
    return entity_words


def load_model(model_id, gpu_id):
    print(f"Loading tokenizer: {model_id}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model: {model_id}", flush=True)
    print("Using 4-bit BitsAndBytes quantization.", flush=True)
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quant_config,
        device_map={"": gpu_id},
    )
    model.generation_config.temperature = None
    model.generation_config.top_p = None

    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=128,
        do_sample=False,
        return_full_text=False,
    )
    print("Model loaded successfully.", flush=True)
    return generator


def make_chat(prompt):
    return [
        {
            "role": "system",
            "content": (
                "You are a precise translation engine. "
                "Return only the translated sentence. Do not explain."
            ),
        },
        {"role": "user", "content": prompt},
    ]


def extract_generation_text(result):
    if isinstance(result, list):
        if not result:
            return ""
        if isinstance(result[-1], dict) and "generated_text" in result[-1]:
            text = result[-1]["generated_text"]
        elif isinstance(result[-1], dict) and "content" in result[-1]:
            return str(result[-1]["content"]).strip()
        else:
            return str(result[-1]).strip()
    else:
        text = result["generated_text"]

    if isinstance(text, list):
        if not text:
            return ""
        if isinstance(text[-1], dict) and "content" in text[-1]:
            return str(text[-1]["content"]).strip()
        return str(text[-1]).strip()
    return str(text).strip()


def ask_llama_batch(generator, prompts, batch_size):
    chats = [make_chat(prompt) for prompt in prompts]
    results = generator(chats, batch_size=batch_size)
    return [extract_generation_text(result) for result in results]


def round_trip_sentence_all_languages(generator, sentence, batch_size):
    sentence = remove_emojis(sentence).strip()
    forward_prompts = [
        f"Translate this English sentence to {language_name}: {sentence}"
        for _, language_name in LANGUAGES
    ]
    translated_sentences = ask_llama_batch(generator, forward_prompts, batch_size)
    backward_prompts = [
        f"Translate this {language_name} sentence to English: {translated_sentence}"
        for (_, language_name), translated_sentence in zip(LANGUAGES, translated_sentences)
    ]
    back_to_english_sentences = ask_llama_batch(generator, backward_prompts, batch_size)
    return {
        language_key: {
            "translated_sentence": translated_sentence,
            "back_to_english_sentence": back_to_english,
        }
        for (language_key, _), translated_sentence, back_to_english in zip(
            LANGUAGES,
            translated_sentences,
            back_to_english_sentences,
        )
    }

def best_back_translation_match(candidate_norm, back_words):
    if not back_words:
        return "", 0
    scored = [(back_word, fuzz.ratio(candidate_norm, back_word)) for back_word in back_words]
    return max(scored, key=lambda item: item[1])


def score_sentence_words(row, generator, fuzzy_threshold, batch_size, nlp):
    sentence = remove_emojis(row["sentence"])
    language_results = round_trip_sentence_all_languages(generator, sentence, batch_size)

    original_words = tokenize_words(sentence)
    original_word_set = normalized_word_set(sentence)
    entity_word_set = named_entity_word_set(sentence, nlp)
    translated_word_sets_by_language = {
        language_key: normalized_word_set(result["translated_sentence"])
        for language_key, result in language_results.items()
    }
    back_word_sets_by_language = {
        language_key: normalized_word_set(result["back_to_english_sentence"])
        for language_key, result in language_results.items()
    }
    back_words_by_language = {
        language_key: [
            normalize_word(word)
            for word in tokenize_words(result["back_to_english_sentence"])
            if normalize_word(word)
        ]
        for language_key, result in language_results.items()
    }

    rows = []
    seen_positions = set()
    for word_index, original_word in enumerate(original_words):
        candidate_norm = normalize_word(original_word)
        if not candidate_norm:
            continue

        position_key = (word_index, candidate_norm)
        if position_key in seen_positions:
            continue
        seen_positions.add(position_key)

        output_row = {
            "sentence_id": row["sentence_id"],
            "sentence": sentence,
            "candidate": original_word,
            "cand_norm": candidate_norm,
            "cand_type": "is_single_word",
            "is_gold": is_gold_candidate(candidate_norm, row["gold_word_norms"]),
            "token_pos": word_index,
            "is_named_entity": int(candidate_norm in entity_word_set),
            "gold_terms": row["gold_terms"],
            "source_dataset": row["source_dataset"],
            "term_type": row["term_type"],
        }

        specialized_votes = 0
        for language_key, _ in LANGUAGES:
            best_match, best_ratio = best_back_translation_match(
                candidate_norm,
                back_words_by_language[language_key],
            )
            language_specialized = int(best_ratio < fuzzy_threshold)
            term_failed_to_translate = (
                candidate_norm in original_word_set
                and candidate_norm in translated_word_sets_by_language[language_key]
                and candidate_norm in back_word_sets_by_language[language_key]
            )
            if term_failed_to_translate and candidate_norm not in entity_word_set:
                language_specialized = 1

            specialized_votes += language_specialized
            output_row[f"{language_key}_sentence"] = language_results[language_key]["translated_sentence"]
            output_row[f"{language_key}_back_to_english_sentence"] = language_results[language_key]["back_to_english_sentence"]
            output_row[f"{language_key}_best_back_translation_word"] = best_match
            output_row[f"{language_key}_fuzz_ratio"] = best_ratio
            output_row[f"{language_key}_specialized"] = language_specialized
            output_row[f"{language_key}_term_failed_to_translate"] = int(term_failed_to_translate)

        output_row["specialized_votes"] = specialized_votes
        output_row["normal_votes"] = len(LANGUAGES) - specialized_votes
        rows.append(output_row)
    return rows


def normalize_scores(values):
    values = pd.to_numeric(values, errors="raise").astype(float)
    total_languages = len(LANGUAGES)

    if total_languages <= 0:
        raise ValueError("LANGUAGES must contain at least one language.")

    invalid = (values < 0) | (values > total_languages)
    if invalid.any():
        invalid_values = sorted(values.loc[invalid].unique().tolist())
        raise ValueError(
            f"specialized_votes must be between 0 and {total_languages}; "
            f"found {invalid_values}"
        )

    return values / float(total_languages)


def build_thresholds(scores, num_thresholds):
    scores = np.asarray(scores, dtype=float)
    scores = scores[np.isfinite(scores)]
    if len(scores) == 0:
        return np.array([])
    unique_scores = np.unique(scores)
    if len(unique_scores) <= num_thresholds:
        return np.round(unique_scores, 6)
    return np.round(np.unique(np.quantile(scores, np.linspace(0, 1, num_thresholds))), 6)


def evaluate_thresholds(candidates_df, num_thresholds):
    y_true = candidates_df["is_gold"].astype(int).values
    scores = candidates_df["mt_score"].astype(float).values
    rows = []
    for threshold in build_thresholds(scores, num_thresholds):
        y_pred = (scores >= threshold).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            average="binary",
            zero_division=0,
        )
        rows.append(
            {
                "candidate_type": "is_single_word",
                "score_column": "mt_score",
                "threshold": threshold,
                "precision": precision,
                "recall": recall,
                "f1": f1,
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
          kind="mergesort",
      ).index[0]


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Starting MT judge run.", flush=True)
    print(f"Dataset: {args.dataset}", flush=True)
    print(f"Output directory: {args.output_dir}", flush=True)

    print(f"Loading spaCy NER model: {args.spacy_model}", flush=True)
    nlp = spacy.load(args.spacy_model)

    generator = load_model(args.model_id, args.gpu_id)

    dataset_df = load_normalized_dataset(args.dataset, args.limit_rows)
    print(f"Loaded {len(dataset_df)} unique sentences.", flush=True)

    all_rows = []
    for idx, row in dataset_df.iterrows():
        print(f"Processing sentence {idx + 1}/{len(dataset_df)}", flush=True)
        all_rows.extend(
            score_sentence_words(
                row,
                generator,
                args.fuzzy_threshold,
                args.batch_size,
                nlp,
            )
        )

    candidates_df = pd.DataFrame(all_rows)
    if candidates_df.empty:
        raise ValueError("No word candidates were produced.")

    candidates_df["mt_score"] = normalize_scores(candidates_df["specialized_votes"])
    scores_path = args.output_dir / "_mt_word_scores_tmp.csv"
    candidates_df.to_csv(scores_path, index=False, encoding="utf-8-sig")
    print(f"Wrote {len(candidates_df)} word scores to {scores_path}", flush=True)

    threshold_df = evaluate_thresholds(candidates_df, args.num_thresholds)
    threshold_path = args.output_dir / "thr_metrics.csv"
    threshold_df.to_csv(threshold_path, index=False)
    print(threshold_df.to_string(index=False), flush=True)
    selected = threshold_df[threshold_df["selected_for_prediction"].eq(1)]
    if not selected.empty:
        row = selected.iloc[0]
        print(
            f"Best threshold for prediction: {row['threshold']} "
            f"(f1={row['f1']:.4f})",
            flush=True,
        )
    print(f"Wrote threshold metrics to {threshold_path}", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
