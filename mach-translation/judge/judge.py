# -*- coding: utf-8 -*-

"""
Candidate Detection: Machine Translation
Judge: Majority vote based, ties are counted as specialized
Languages: Spanish, Arabic, Chinese, Japanese
Fuzz Threshold: 93
"""

print("Starting judge script...", flush=True)

import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import regex as re
import spacy
import torch
from datasets import load_dataset
from rapidfuzz import fuzz
from sklearn.metrics import precision_recall_fscore_support
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

print("Imports complete", flush=True)

MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
DATASET_ID = "acader/genz-alpha-slangs"
OUTPUT_DIR = Path("detect3_judge_outputs")

# Judge system of languages
LANGUAGES = [
    ("spanish", "Spanish"),
    ("arabic", "Arabic"),
    ("chinese", "Chinese"),
    ("japanese", "Japanese"),
]

WORD_RE = re.compile(r"[\p{L}\p{M}]+(?:['\u2019-][\p{L}\p{M}]+)?|[\p{N}]+")
EMOJI_RE = re.compile(r"\p{Emoji_Presentation}|\p{Extended_Pictographic}")

# Helper functions to clean texts
def remove_emojis(text):
    return EMOJI_RE.sub("", str(text))


def normalize_text(text):
    text = remove_emojis(text)
    text = str(text).lower().strip()
    text = re.sub(r"^[^\p{L}\p{N}]+|[^\p{L}\p{N}]+$", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_word(text):
    text = remove_emojis(text)
    text = str(text).lower().strip()
    text = re.sub(r"[^\p{L}\p{N}'\u2019-]", "", text)
    return text


def tokenize_words(text):
    text = remove_emojis(text)
    return [m.group(0) for m in WORD_RE.finditer(str(text))]


def normalized_word_set(text):
    words = [normalize_word(w) for w in tokenize_words(text)]
    return {w for w in words if w}


# Detects false flags confusing specialized terms with named entities
def named_entity_word_set(sentence, nlp):
    doc = nlp(str(sentence))
    entity_words = set()

    for ent in doc.ents:
        for word in tokenize_words(ent.text):
            word_norm = normalize_word(word)
            if word_norm:
                entity_words.add(word_norm)

    return entity_words


# Helper functions for gold labels
def extract_gold_terms(explanation):
    if not isinstance(explanation, str):
        return []

    terms = re.findall(r'\*\*"([^"]+)"\*\*:', explanation)
    if not terms:
        terms = re.findall(r'"([^"]+)"\s*:', explanation)

    return [remove_emojis(t).strip() for t in terms if remove_emojis(t).strip()]


def is_gold_candidate(candidate_norm, gold_norms):
    for gold_norm in gold_norms:
        if not gold_norm:
            continue
        if candidate_norm == gold_norm:
            return 1
        if fuzz.ratio(candidate_norm, gold_norm) >= 92:
            return 1
    return 0


def expand_gold_terms_to_words(gold_terms):
    gold_words = []

    for term in gold_terms:
        for word in tokenize_words(term):
            word_norm = normalize_word(word)
            if word_norm:
                gold_words.append(word_norm)

    return gold_words


# Load model
def load_translation_pipeline():
    print("Loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Use GPU - expected runtime ~ 1 hour
    use_cuda = torch.cuda.is_available()
    print("CUDA available:", use_cuda, flush=True)
    if use_cuda:
        print("GPU:", torch.cuda.get_device_name(0), flush=True)

    dtype = torch.bfloat16 if use_cuda else torch.float32

    print("Loading model...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        device_map="auto",
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

    print("Model loaded successfully", flush=True)
    return generator


# Prompts model in chat
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


# Pulls clean output
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


# Runs several translation prompts in a batch
def ask_llama_batch(generator, prompts, batch_size):
    chats = [make_chat(prompt) for prompt in prompts]
    results = generator(chats, batch_size=batch_size)
    return [extract_generation_text(result) for result in results]


# Runs 4 languages in parallel
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


"""
For one original word, search for the most similar word in the final translation. This will indicate if the term disappeared or changed enough to be considered specialized.
"""
def best_back_translation_match(candidate_norm, back_words):
    if not back_words:
        return "", 0

    scored = [(back_word, fuzz.ratio(candidate_norm, back_word)) for back_word in back_words]
    return max(scored, key=lambda x: x[1])


# Score every word per one sentence
def score_sentence_words(row, generator, fuzzy_threshold, batch_size, nlp):
    sentence = remove_emojis(row["Sentence"])
    
    # Translate sentence and return results
    language_results = round_trip_sentence_all_languages(generator, sentence, batch_size)

    original_words = tokenize_words(sentence)
    original_word_set = normalized_word_set(sentence)
    entity_word_set = named_entity_word_set(sentence, nlp) # Save named entities
    translated_word_sets_by_language = {
        language_key: normalized_word_set(result["translated_sentence"])
        for language_key, result in language_results.items()
    }
    back_word_sets_by_language = {
        language_key: normalized_word_set(result["back_to_english_sentence"])
        for language_key, result in language_results.items()
    }
    back_words_by_language = {}

    for language_key, result in language_results.items():
        back_words = [
            normalize_word(w)
            for w in tokenize_words(result["back_to_english_sentence"])
        ]
        back_words_by_language[language_key] = [w for w in back_words if w]

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
            "doc_id": row["doc_id"],
            "sentence": sentence,
            "candidate": original_word,
            "candidate_norm": candidate_norm,
            "is_gold": is_gold_candidate(candidate_norm, row["gold_word_norms"]),
            "is_named_entity": int(candidate_norm in entity_word_set),
            "gold_terms": "; ".join(row["gold_terms"]),
        }

        specialized_votes = 0

        # For each language provide results
        for language_key, _ in LANGUAGES:
            best_match, best_ratio = best_back_translation_match(
                candidate_norm,
                back_words_by_language[language_key],
            )
            # Fuzzy threshold comparison
            language_specialized = int(best_ratio < fuzzy_threshold)
            
            # NER comparison - checks for terms that failed to translate
            language_term_failed_to_translate = (
                candidate_norm in original_word_set
                and candidate_norm in translated_word_sets_by_language[language_key]
                and candidate_norm in back_word_sets_by_language[language_key]
            )
            is_named_entity = int(candidate_norm in entity_word_set)

            if language_term_failed_to_translate and not is_named_entity:
                language_specialized = 1

            specialized_votes += language_specialized

            output_row[f"{language_key}_sentence"] = language_results[language_key][
                "translated_sentence"
            ]
            output_row[f"{language_key}_back_to_english_sentence"] = language_results[
                language_key
            ]["back_to_english_sentence"]
            output_row[f"{language_key}_best_back_translation_word"] = best_match
            output_row[f"{language_key}_fuzz_ratio"] = best_ratio
            output_row[f"{language_key}_specialized"] = language_specialized
            output_row[f"{language_key}_term_failed_to_translate"] = int(
                language_term_failed_to_translate
            )
        # Tally votes - 2 or more votes labels word as specialized
        output_row["specialized_votes"] = specialized_votes
        output_row["normal_votes"] = len(LANGUAGES) - specialized_votes
        output_row["specialized"] = int(specialized_votes >= 2)

        rows.append(output_row)

    return rows

# Helper functions to evaluate results
def evaluate_results(all_candidates_df):
    y_true = all_candidates_df["is_gold"].astype(int).values
    y_pred = all_candidates_df["specialized"].astype(int).values

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        zero_division=0,
    )

    return pd.DataFrame([{
        "comparison": "four_language_judge_vs_gold_terms",
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "predicted_count": int(y_pred.sum()),
        "gold_count": int(y_true.sum()),
        "candidate_count": int(len(y_true)),
    }])


def confusion_counts(all_candidates_df):
    y_true = all_candidates_df["is_gold"].astype(int)
    y_pred = all_candidates_df["specialized"].astype(int)

    true_positives = int(((y_true == 1) & (y_pred == 1)).sum())
    false_positives = int(((y_true == 0) & (y_pred == 1)).sum())
    false_negatives = int(((y_true == 1) & (y_pred == 0)).sum())
    true_negatives = int(((y_true == 0) & (y_pred == 0)).sum())

    return pd.DataFrame([{
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "true_negatives": true_negatives,
    }])


def plot_metrics(metrics_df, output_dir):
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    row = metrics_df.iloc[0]
    labels = ["precision", "recall", "f1"]
    values = [row["precision"], row["recall"], row["f1"]]

    plt.figure(figsize=(6, 4))
    plt.bar(labels, values)
    plt.ylim(0, 1)
    plt.ylabel("Metric")
    plt.title("Method 3 Judge: Four-Language Vote")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    output_path = plot_dir / "detect3_judge_precision_recall_f1.png"
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {output_path}", flush=True)


def plot_confusion_counts(counts_df, output_dir):
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    row = counts_df.iloc[0]
    labels = ["true positives", "false positives", "false negatives"]
    values = [
        row["true_positives"],
        row["false_positives"],
        row["false_negatives"],
    ]

    plt.figure(figsize=(7, 4))
    plt.bar(labels, values)
    plt.ylabel("Count")
    plt.title("Method 3 Judge: TP, FP, and FN Counts")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    output_path = plot_dir / "detect3_judge_tp_fp_fn_counts.png"
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {output_path}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Detect slang candidates using a four-language round-trip translation judge."
    )
    parser.add_argument("--limit-rows", type=int, default=None, help="Optional small dataset slice for testing.")
    parser.add_argument("--fuzzy-threshold", type=int, default=93, help="Flag words whose best fuzzy match is below this.")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for the four translation prompts.")
    parser.add_argument("--spacy-model", default="en_core_web_sm", help="spaCy model used for named entity recognition.")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset and extract gold labels
    print(f"Loading dataset: {DATASET_ID}", flush=True)
    dataset = load_dataset(DATASET_ID)
    df = pd.DataFrame(dataset["train"]).reset_index(drop=True)

    required_cols = {"Sentence", "Explanation"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if args.limit_rows is not None:
        df = df.head(args.limit_rows).copy()

    df["doc_id"] = df.index
    df["Sentence"] = df["Sentence"].apply(remove_emojis)
    df["gold_terms"] = df["Explanation"].apply(extract_gold_terms)
    df["gold_norms"] = df["gold_terms"].apply(
        lambda xs: [normalize_text(x) for x in xs if normalize_text(x)]
    )
    df["gold_word_norms"] = df["gold_terms"].apply(expand_gold_terms_to_words)

    # Load spaCy
    print(f"Loading spaCy NER model: {args.spacy_model}", flush=True)
    try:
        nlp = spacy.load(args.spacy_model)
    except OSError as exc:
        raise OSError(
            f"Could not load spaCy model '{args.spacy_model}'. Install it with: "
            f"python3 -m spacy download {args.spacy_model}"
        ) from exc

    # Load LLama
    generator = load_translation_pipeline()

    # Run machine translation
    all_rows = []
    for i, row in df.iterrows():
        print(f"Processing sentence {i + 1}/{len(df)}", flush=True) # Progress update
        all_rows.extend(
            score_sentence_words(
                row,
                generator,
                args.fuzzy_threshold,
                args.batch_size,
                nlp,
            )
        )

    # Return various results
    all_candidates_df = pd.DataFrame(all_rows)
    flagged_df = all_candidates_df[all_candidates_df["specialized"] == 1].copy()

    all_candidates_path = output_dir / "detect3_judge_all_word_scores.csv"
    flagged_path = output_dir / "detect3_judge_flagged_candidates.csv"
    metrics_path = output_dir / "detect3_judge_precision_recall_f1.csv"
    counts_path = output_dir / "detect3_judge_tp_fp_fn_counts.csv"

    all_candidates_df.to_csv(
        all_candidates_path,
        index=False,
        float_format="%.2f",
        encoding="utf-8-sig",
    )
    flagged_df.to_csv(
        flagged_path,
        index=False,
        float_format="%.2f",
        encoding="utf-8-sig",
    )

    metrics_df = evaluate_results(all_candidates_df)
    metrics_df.to_csv(
        metrics_path,
        index=False,
        float_format="%.2f",
        encoding="utf-8-sig",
    )
    plot_metrics(metrics_df, output_dir)

    counts_df = confusion_counts(all_candidates_df)
    counts_df.to_csv(
        counts_path,
        index=False,
        float_format="%.2f",
        encoding="utf-8-sig",
    )
    plot_confusion_counts(counts_df, output_dir)

    print(f"Saved all word scores: {all_candidates_path}", flush=True)
    print(f"Saved flagged candidates: {flagged_path}", flush=True)
    print(f"Saved metrics: {metrics_path}", flush=True)
    print(f"Saved TP/FP/FN counts: {counts_path}", flush=True)
    print(metrics_df, flush=True)
    print(counts_df, flush=True)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
