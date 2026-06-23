# -*- coding: utf-8 -*-

"""
Candidate Detection: Machine Translation
Method: Layers - translates original sentence through pipeline of languages
Languages (in order): EN -> ES -> AR -> CHS -> JPN -> EN
Fuzz Threshold: 93
"""
print("Starting layered translation script...", flush=True)

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
OUTPUT_DIR = Path("layers_outputs")

WORD_RE = re.compile(r"[\p{L}\p{M}]+(?:['\u2019-][\p{L}\p{M}]+)?|[\p{N}]+")
EMOJI_RE = re.compile(r"\p{Emoji_Presentation}|\p{Extended_Pictographic}")


# Helper functions for clean text
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


# Detects named entities
def entity_word_sets(sentences, nlp):
    results = []

    for doc in nlp.pipe([str(s) for s in sentences], batch_size=64):
        entity_words = set()
        for ent in doc.ents:
            for word in tokenize_words(ent.text):
                word_norm = normalize_word(word)
                if word_norm:
                    entity_words.add(word_norm)
        results.append(entity_words)

    return results


# Helper functions for extracting gold terms
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


# Loads model
def load_translation_pipeline(max_new_tokens):
    print("Loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Use GPU - expected runtime ~50 min
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
        max_new_tokens=max_new_tokens,
        do_sample=False,
        return_full_text=False,
    )

    print("Model loaded successfully", flush=True)
    return generator


# Prompts model through chat
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


# Extracts plain text from model output
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


# Runs multiple prompts to model
def ask_llama_batch(generator, prompts, generation_batch_size):
    chats = [make_chat(prompt) for prompt in prompts]
    results = generator(chats, batch_size=generation_batch_size)
    return [extract_generation_text(result) for result in results]


# Layered translation
def translate_layer(generator, source_language, target_language, sentences, generation_batch_size):
    prompts = [
        f"Translate this {source_language} sentence to {target_language}: {sentence}"
        for sentence in sentences
    ]
    return ask_llama_batch(generator, prompts, generation_batch_size)


# Chain of translations
def layered_round_trip_batch(generator, english_sentences, generation_batch_size):
    english_sentences = [remove_emojis(s).strip() for s in english_sentences]

    spanish_sentences = translate_layer(
        generator, "English", "Spanish", english_sentences, generation_batch_size
    )
    arabic_sentences = translate_layer(
        generator, "Spanish", "Arabic", spanish_sentences, generation_batch_size
    )
    chinese_sentences = translate_layer(
        generator, "Arabic", "Chinese", arabic_sentences, generation_batch_size
    )
    japanese_sentences = translate_layer(
        generator, "Chinese", "Japanese", chinese_sentences, generation_batch_size
    )
    back_to_english_sentences = translate_layer(
        generator, "Japanese", "English", japanese_sentences, generation_batch_size
    )

    return pd.DataFrame({
        "sentence": english_sentences,
        "spanish_sentence": spanish_sentences,
        "arabic_sentence": arabic_sentences,
        "chinese_sentence": chinese_sentences,
        "japanese_sentence": japanese_sentences,
        "back_to_english_sentence": back_to_english_sentences,
    })


# Searches for most similar word in final translation to original sentence
def best_back_translation_match(candidate_norm, back_words):
    if not back_words:
        return "", 0

    scored = [(back_word, fuzz.ratio(candidate_norm, back_word)) for back_word in back_words]
    return max(scored, key=lambda x: x[1])


# Score candidates
def score_translated_batch(batch_df, translations_df, nlp, fuzzy_threshold):
    all_rows = []
    entity_sets = entity_word_sets(translations_df["sentence"].tolist(), nlp)

    for offset, (_, row) in enumerate(batch_df.iterrows()):
        translation_row = translations_df.iloc[offset]
        sentence = translation_row["sentence"]
        back_to_english = translation_row["back_to_english_sentence"]

        original_words = tokenize_words(sentence)
        back_words = [normalize_word(w) for w in tokenize_words(back_to_english)]
        back_words = [w for w in back_words if w]

        translated_sets = {
            "spanish": normalized_word_set(translation_row["spanish_sentence"]),
            "arabic": normalized_word_set(translation_row["arabic_sentence"]),
            "chinese": normalized_word_set(translation_row["chinese_sentence"]),
            "japanese": normalized_word_set(translation_row["japanese_sentence"]),
        }
        entity_word_set = entity_sets[offset]

        seen_positions = set()
        for word_index, original_word in enumerate(original_words):
            candidate_norm = normalize_word(original_word)
            if not candidate_norm:
                continue

            position_key = (word_index, candidate_norm)
            if position_key in seen_positions:
                continue
            seen_positions.add(position_key)

            # Scores by similarity
            best_match, best_ratio = best_back_translation_match(candidate_norm, back_words)
            specialized = int(best_ratio < fuzzy_threshold)

            # Check for named entity false flagging
            spanish_term_unchanged = candidate_norm in translated_sets["spanish"]
            arabic_term_unchanged = candidate_norm in translated_sets["arabic"]
            chinese_term_unchanged = candidate_norm in translated_sets["chinese"]
            japanese_term_unchanged = candidate_norm in translated_sets["japanese"]
            term_unchanged_in_any_translation = any([
                spanish_term_unchanged,
                arabic_term_unchanged,
                chinese_term_unchanged,
                japanese_term_unchanged,
            ])
            is_named_entity = int(candidate_norm in entity_word_set)

            if term_unchanged_in_any_translation and not is_named_entity:
                specialized = 1

            all_rows.append({
                "doc_id": row["doc_id"],
                "sentence": sentence,
                "spanish_sentence": translation_row["spanish_sentence"],
                "arabic_sentence": translation_row["arabic_sentence"],
                "chinese_sentence": translation_row["chinese_sentence"],
                "japanese_sentence": translation_row["japanese_sentence"],
                "back_to_english_sentence": back_to_english,
                "candidate": original_word,
                "candidate_norm": candidate_norm,
                "best_back_translation_word": best_match,
                "fuzz_ratio": best_ratio,
                "specialized": specialized,
                "term_unchanged_in_any_translation": int(term_unchanged_in_any_translation),
                "spanish_term_unchanged": int(spanish_term_unchanged),
                "arabic_term_unchanged": int(arabic_term_unchanged),
                "chinese_term_unchanged": int(chinese_term_unchanged),
                "japanese_term_unchanged": int(japanese_term_unchanged),
                "is_named_entity": is_named_entity,
                "is_gold": is_gold_candidate(candidate_norm, row["gold_word_norms"]),
                "gold_terms": "; ".join(row["gold_terms"]),
            })

    return all_rows


# Helper functions for results
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
        "comparison": "layered_translation_vs_gold_terms",
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
    plt.title("Layered Translation: Precision, Recall, F1")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    output_path = plot_dir / "layers_precision_recall_f1.png"
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
    plt.title("Layered Translation: TP, FP, and FN Counts")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    output_path = plot_dir / "layers_tp_fp_fn_counts.png"
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {output_path}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Detect slang candidates using layered English-Spanish-Arabic-Chinese-Japanese-English translation."
    )
    parser.add_argument("--limit-rows", type=int, default=None, help="Optional small dataset slice for testing.")
    parser.add_argument("--fuzzy-threshold", type=int, default=93, help="Flag words whose best fuzzy match is below this.")
    parser.add_argument("--row-batch-size", type=int, default=8, help="Number of dataset rows translated through the layer chain at once.")
    parser.add_argument("--generation-batch-size", type=int, default=4, help="Batch size passed to the Hugging Face generation pipeline.")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--spacy-model", default="en_core_web_sm", help="spaCy model used for named entity recognition.")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset and save gold labels
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

    print(f"Loading spaCy NER model: {args.spacy_model}", flush=True)
    try:
        nlp = spacy.load(args.spacy_model)
    except OSError as exc:
        raise OSError(
            f"Could not load spaCy model '{args.spacy_model}'. Install it with: "
            f"python3 -m spacy download {args.spacy_model}"
        ) from exc

    generator = load_translation_pipeline(args.max_new_tokens)

    all_candidate_rows = []
    all_translation_rows = []

    # Run translation pipeline
    for start_idx in range(0, len(df), args.row_batch_size):
        end_idx = min(start_idx + args.row_batch_size, len(df))
        print(f"Processing rows {start_idx + 1}-{end_idx}/{len(df)}", flush=True) # Progress update

        batch_df = df.iloc[start_idx:end_idx].copy()
        translations_df = layered_round_trip_batch(
            generator,
            batch_df["Sentence"].tolist(),
            args.generation_batch_size,
        )
        translations_df.insert(0, "doc_id", batch_df["doc_id"].tolist())
        all_translation_rows.append(translations_df)

        all_candidate_rows.extend(
            score_translated_batch(
                batch_df,
                translations_df,
                nlp,
                args.fuzzy_threshold,
            )
        )

    translations_all_df = pd.concat(all_translation_rows, ignore_index=True)
    all_candidates_df = pd.DataFrame(all_candidate_rows)
    flagged_df = all_candidates_df[all_candidates_df["specialized"] == 1].copy()

    translations_path = output_dir / "layers_sentence_translations.csv"
    all_candidates_path = output_dir / "layers_all_word_scores.csv"
    flagged_path = output_dir / "layers_flagged_candidates.csv"
    metrics_path = output_dir / "layers_precision_recall_f1.csv"
    counts_path = output_dir / "layers_tp_fp_fn_counts.csv"

    translations_all_df.to_csv(
        translations_path,
        index=False,
        float_format="%.3f",
        encoding="utf-8-sig",
    )
    all_candidates_df.to_csv(
        all_candidates_path,
        index=False,
        float_format="%.3f",
        encoding="utf-8-sig",
    )
    flagged_df.to_csv(
        flagged_path,
        index=False,
        float_format="%.3f",
        encoding="utf-8-sig",
    )

    metrics_df = evaluate_results(all_candidates_df)
    metrics_df.to_csv(
        metrics_path,
        index=False,
        float_format="%.3f",
        encoding="utf-8-sig",
    )
    plot_metrics(metrics_df, output_dir)

    counts_df = confusion_counts(all_candidates_df)
    counts_df.to_csv(
        counts_path,
        index=False,
        float_format="%.3f",
        encoding="utf-8-sig",
    )
    plot_confusion_counts(counts_df, output_dir)

    print(f"Saved sentence translations: {translations_path}", flush=True)
    print(f"Saved all word scores: {all_candidates_path}", flush=True)
    print(f"Saved flagged candidates: {flagged_path}", flush=True)
    print(f"Saved metrics: {metrics_path}", flush=True)
    print(f"Saved TP/FP/FN counts: {counts_path}", flush=True)
    print(metrics_df, flush=True)
    print(counts_df, flush=True)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
