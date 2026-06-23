# -*- coding: utf-8 -*-

# Candidate Detection: Machine Translation
# LANGUAGE: JAPANESE
# FUZZ THR: 93
print("Starting script...")

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

print("Imports complete")

MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
DATASET_ID = "acader/genz-alpha-slangs"
OUTPUT_DIR = Path("detect3_outputs")

WORD_RE = re.compile(r"[\p{L}\p{M}]+(?:['\u2019-][\p{L}\p{M}]+)?|[\p{N}]+")
EMOJI_RE = re.compile(r"\p{Emoji_Presentation}|\p{Extended_Pictographic}")


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


# Use for tokenizing phrases into words
def tokenize_words(text):
    text = remove_emojis(text)
    return [m.group(0) for m in WORD_RE.finditer(str(text))]


def normalized_word_set(text):
    words = [normalize_word(w) for w in tokenize_words(text)]
    return {w for w in words if w}


def named_entity_word_set(sentence, nlp):
    doc = nlp(str(sentence))
    entity_words = set()

    for ent in doc.ents:
        for word in tokenize_words(ent.text):
            word_norm = normalize_word(word)
            if word_norm:
                entity_words.add(word_norm)

    return entity_words


def extract_gold_terms(explanation):
    """Extract labeled slang terms from the dataset Explanation column."""
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


# Converts gold phrases to gold words
def expand_gold_terms_to_words(gold_terms):
    gold_words = []

    for term in gold_terms:
        for word in tokenize_words(term):
            word_norm = normalize_word(word)
            if word_norm:
                gold_words.append(word_norm)

    return gold_words


def load_translation_pipeline():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Use GPU
    use_cuda = torch.cuda.is_available()
    print("CUDA available:", use_cuda)
    if use_cuda:
        print("GPU:", torch.cuda.get_device_name(0))

    dtype = torch.bfloat16 if use_cuda else torch.float32

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        device_map="auto",
    )

    # Text generation pipeline
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=128, # Max response length
        temperature=0.0,
        do_sample=False,
        return_full_text=False,
    )

    print("Model loaded successfully")
    return generator


# Prompts Llama, Run Inference
def ask_llama(generator, prompt):
    messages = [
        {
            "role": "system",
            "content": (
                "You are a precise translation engine. "
                "Return only the translated sentence. Do not explain."
            ),
        },
        {"role": "user", "content": prompt},
    ]

    result = generator(messages)[0]["generated_text"]

    # If model returns chat msg format, take only the last message content
    if isinstance(result, list):
        return result[-1]["content"].strip()

    return str(result).strip()


# Machine Translation
def round_trip_sentence(generator, sentence):
    sentence = remove_emojis(sentence).strip()

    # Runs inference two separate times to avoid cheating
    japanese_sentence = ask_llama(
        generator,
        f"Translate this English sentence to Japanese: {sentence}",
    )

    back_to_english = ask_llama(
        generator,
        f"Translate this Japanese sentence to English: {japanese_sentence}",
    )

    return japanese_sentence, back_to_english


# Compares initial and final sentence matches to see if words changed/disappeared
def best_back_translation_match(candidate_norm, back_words):
    if not back_words:
        return "", 0

    scored = [(back_word, fuzz.ratio(candidate_norm, back_word)) for back_word in back_words]
    return max(scored, key=lambda x: x[1])


# Score each word in one sentence
def score_sentence_words(row, generator, fuzzy_threshold, nlp):

    # Sentence input
    sentence = remove_emojis(row["Sentence"])
    
    # Machine translation
    japanese_sentence, back_to_english = round_trip_sentence(generator, sentence)

    original_words = tokenize_words(sentence)
    back_words = [normalize_word(w) for w in tokenize_words(back_to_english)]
    back_words = [w for w in back_words if w]
    original_word_set = normalized_word_set(sentence)
    translated_word_set = normalized_word_set(japanese_sentence)
    back_word_set = normalized_word_set(back_to_english)
    entity_word_set = named_entity_word_set(sentence, nlp)

    rows = []
    seen_positions = set()

    # Scoring
    for word_index, original_word in enumerate(original_words):
        candidate_norm = normalize_word(original_word)
        if not candidate_norm:
            continue

        # Keep repeated words as separate occurrences if they appear at different positions.
        position_key = (word_index, candidate_norm)
        if position_key in seen_positions:
            continue
        seen_positions.add(position_key)

        best_match, best_ratio = best_back_translation_match(candidate_norm, back_words)
        
        # Flag candidates whos best match in final sentence is below THR
        specialized = int(best_ratio < fuzzy_threshold)
        term_failed_to_translate = (
            candidate_norm in original_word_set
            and candidate_norm in translated_word_set
            and candidate_norm in back_word_set
        )
        is_named_entity = int(candidate_norm in entity_word_set)

        if term_failed_to_translate and not is_named_entity:
            specialized = 1

        # Save candidate information
        rows.append({
            "doc_id": row["doc_id"],
            "sentence": sentence,
            "japanese_sentence": japanese_sentence,
            "back_to_english_sentence": back_to_english,
            "candidate": original_word,
            "candidate_norm": candidate_norm,
            "best_back_translation_word": best_match,
            "fuzz_ratio": best_ratio,
            "specialized": specialized,
            "term_failed_to_translate": int(term_failed_to_translate),
            "is_named_entity": is_named_entity,
            "is_gold": is_gold_candidate(candidate_norm, row["gold_word_norms"]),
            "gold_terms": "; ".join(row["gold_terms"]),
        })

    return rows


# Evaluation metrics
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
        "comparison": "sentence_round_trip_flagged_words_vs_gold_terms",
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "predicted_count": int(y_pred.sum()),
        "gold_count": int(y_true.sum()),
        "candidate_count": int(len(y_true)),
    }])


# Computes TP, FP, FN, TN
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
    plt.title("Method 3: Full Sentence Round Trip")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    output_path = plot_dir / "detect3_precision_recall_f1.png"
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {output_path}")


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
    plt.title("Method 3: TP, FP, and FN Counts")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    output_path = plot_dir / "detect3_tp_fp_fn_counts.png"
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {output_path}")


# Command line arguments
def parse_args():
    parser = argparse.ArgumentParser(
        description="Detect slang candidates using full-sentence English-Japanese-English round-trip translation."
    )
    parser.add_argument("--limit-rows", type=int, default=None, help="Optional small dataset slice for testing.")
    parser.add_argument("--fuzzy-threshold", type=int, default=93, help="Flag words whose best fuzzy match is below this.") # Sets fuzz threshold
    parser.add_argument("--spacy-model", default="en_core_web_sm", help="spaCy model used for named entity recognition.")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set up outputs
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    print(f"Loading dataset: {DATASET_ID}")
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

    # Load spaCy NER
    print(f"Loading spaCy NER model: {args.spacy_model}")
    try:
        nlp = spacy.load(args.spacy_model)
    except OSError as exc:
        raise OSError(
            f"Could not load spaCy model '{args.spacy_model}'. Install it with: "
            f"python3 -m spacy download {args.spacy_model}"
        ) from exc

    # Load Llama
    generator = load_translation_pipeline()

    all_rows = []
    for i, row in df.iterrows():
        print(f"Processing sentence {i + 1}/{len(df)}") # Status update
        all_rows.extend(score_sentence_words(row, generator, args.fuzzy_threshold, nlp))

    # Save results
    all_candidates_df = pd.DataFrame(all_rows)
    flagged_df = all_candidates_df[all_candidates_df["specialized"] == 1].copy()

    all_candidates_path = output_dir / "detect3_all_word_scores.csv"
    flagged_path = output_dir / "detect3_flagged_candidates.csv"
    metrics_path = output_dir / "detect3_precision_recall_f1.csv"
    counts_path = output_dir / "detect3_tp_fp_fn_counts.csv"

    all_candidates_df.to_csv(all_candidates_path, index=False, float_format="%.3f")
    flagged_df.to_csv(flagged_path, index=False, float_format="%.3f")

    metrics_df = evaluate_results(all_candidates_df)
    metrics_df.to_csv(metrics_path, index=False, float_format="%.3f")
    plot_metrics(metrics_df, output_dir)

    counts_df = confusion_counts(all_candidates_df)
    counts_df.to_csv(counts_path, index=False, float_format="%.3f")
    plot_confusion_counts(counts_df, output_dir)

    print(f"Saved all word scores: {all_candidates_path}")
    print(f"Saved flagged candidates: {flagged_path}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved TP/FP/FN counts: {counts_path}")
    print(metrics_df)
    print(counts_df)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
