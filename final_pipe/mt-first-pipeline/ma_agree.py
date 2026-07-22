# -*- coding: utf-8 -*-

import argparse
import gc
import os
import unicodedata
from pathlib import Path

import pandas as pd
import regex as re
import torch
from rapidfuzz import fuzz
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline


REQUIRED_DATASET_COLUMNS = ["sentence", "gold_terms", "term_type", "source_dataset", "is_single_word", "is_multiword"]
MODEL1 = "meta-llama/Meta-Llama-3.1-8B-Instruct"
MODEL2 = "Qwen/Qwen2.5-7B-Instruct"
MODEL3 = "mistralai/Mistral-7B-Instruct-v0.3"
MODEL_NAMES = ["llama", "qwen", "mistral"]
EMOJI_RE = re.compile(r"\p{Emoji_Presentation}|\p{Extended_Pictographic}")
APOSTROPHES = str.maketrans({"\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201b": "'", "\u2032": "'", "\u2035": "'", "`": "'", "\u00b4": "'"})
HYPHENS = str.maketrans({"\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-", "\u2014": "-", "\u2015": "-", "\u2212": "-"})


def parse_args():
    parser = argparse.ArgumentParser(description="Generate cross-model definitions for MT-proposed candidates.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mt-input", type=Path, default=None)
    parser.add_argument("--definitions-output", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit-rows", type=int, default=None)
    parser.add_argument("--gpu-ids", default="0,0,0")
    parser.add_argument("--model-ids", nargs=3, default=[MODEL1, MODEL2, MODEL3])
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
    text = EMOJI_RE.sub("", str(text))
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(APOSTROPHES).translate(HYPHENS)
    text = text.casefold().strip()
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


def is_gold_candidate(candidate_norm, gold_norms):
    for gold_norm in gold_norms:
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
    df["is_multiword"] = df["is_multiword"].apply(to_binary)
    if limit_rows is not None:
        keep_sentences = df["sentence"].drop_duplicates().head(limit_rows)
        df = df[df["sentence"].isin(keep_sentences)].copy()
    return df


def gold_by_sentence_and_type(dataset_df):
    sentence_ids = {sentence: index for index, sentence in enumerate(dataset_df["sentence"].drop_duplicates().tolist())}
    output = {}
    for sentence, group in dataset_df.groupby("sentence", sort=False):
        sid = str(sentence_ids[sentence])
        output[(sid, "is_single_word")] = {normalize_word(value) for value in group.loc[group["is_single_word"].eq(1), "gold_norm"] if value}
        output[(sid, "is_multiword")] = {normalize_text(value) for value in group.loc[group["is_multiword"].eq(1), "gold_norm"] if value}
    return output


def load_mt_candidates(mt_input, dataset_df):
    mt_df = pd.read_csv(mt_input)
    if "candidate_type" in mt_df.columns and "cand_type" not in mt_df.columns:
        mt_df = mt_df.rename(columns={"candidate_type": "cand_type"})
    require_columns(mt_df, ["sentence_id", "sentence", "candidate", "cand_norm", "cand_type", "mt_pred"], mt_input)
    mt_df = mt_df.copy()
    mt_df["cand_type"] = mt_df["cand_type"].astype(str)
    mt_df["cand_norm"] = mt_df.apply(normalize_candidate, axis=1)
    gold_lookup = gold_by_sentence_and_type(dataset_df)
    mt_df["is_gold"] = mt_df.apply(lambda row: is_gold_candidate(row["cand_norm"], gold_lookup.get((str(row["sentence_id"]), str(row["cand_type"]).lower()), set())), axis=1)
    return mt_df


def parse_gpu_ids(value):
    gpu_ids = [int(item.strip()) for item in str(value).split(",") if item.strip()]
    if len(gpu_ids) != 3:
        raise ValueError("--gpu-ids must provide exactly three comma-separated ids")
    return gpu_ids


def check_cuda_gpus(required_gpus):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Request a GPU in the SLURM job.")
    gpu_count = torch.cuda.device_count()
    if gpu_count < required_gpus:
        raise RuntimeError(f"Need {required_gpus} CUDA GPUs, but only {gpu_count} are visible.")
    print(f"Visible CUDA GPUs: {gpu_count}", flush=True)
    for gpu_id in range(gpu_count):
        print(f"GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}", flush=True)


def load_model(model_id, gpu_id):
    print(f"Loading tokenizer: {model_id}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"Loading model: {model_id}", flush=True)
    quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quant_config, device_map={"": gpu_id})
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    return pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=40, do_sample=False, return_full_text=False)


def unload_model(generator):
    if generator is None:
        return
    try:
        del generator.model
        del generator.tokenizer
    except AttributeError:
        pass
    del generator
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def make_chat(candidate, sentence, cand_type):
    cand_type_norm = str(cand_type).lower()
    if cand_type_norm == "is_multiword":
        return [
            {
                "role": "system",
                "content": (
                    "You are a strict contextual glossary writer. "
                    "Define only the target span's meaning in the given sentence. "
                    "Use the sentence context to choose the correct sense. "
                    "Do not define words not included in the span. "
                    "Do not repeat the sentence. "
                    "Do not give examples. "
                    "Return only the definition. "
                    "Use no more than 20 words."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Sentence: {sentence}\n"
                    f"Target span: {candidate}\n"
                    "Context-specific definition:"
                ),
            },
        ]

    return [
        {
            "role": "system",
            "content": (
                "You are a strict contextual glossary writer. "
                "Define only the target term's meaning in the given sentence. "
                "Use the sentence context to choose the correct sense. "
                "Do not define other words. "
                "Do not repeat the sentence. "
                "Do not give examples. "
                "Return only the definition. "
                "Use no more than 20 words."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Sentence: {sentence}\n"
                f"Target term: {candidate}\n"
                "Context-specific definition:"
            ),
        },
    ]


def extract_generation_text(result):
    if isinstance(result, list):
        if not result:
            return ""
        text = result[-1].get("generated_text", result[-1].get("content", result[-1])) if isinstance(result[-1], dict) else result[-1]
    else:
        text = result["generated_text"]
    if isinstance(text, list):
        if not text:
            return ""
        if isinstance(text[-1], dict) and "content" in text[-1]:
            return str(text[-1]["content"]).strip()
        return str(text[-1]).strip()
    return str(text).strip()


def define_candidates_for_model(definitions_df, model_name, generator, batch_size, output_path):
    definitions_df = definitions_df.copy()
    score_mask = definitions_df["mt_pred"].apply(to_binary).eq(1)
    definitions_df[f"{model_name}_definition"] = ""
    todo_df = definitions_df[score_mask].copy()
    chats = [
        make_chat(row["candidate"], row["sentence"], row["cand_type"])
        for _, row in todo_df.iterrows()
    ]
    definitions = []
    total = len(todo_df)
    print(f"Generating {model_name} definitions for {total} MT-positive candidates...", flush=True)
    for start_idx in range(0, total, batch_size):
        end_idx = min(start_idx + batch_size, total)
        batch_results = generator(chats[start_idx:end_idx], batch_size=batch_size)
        definitions.extend(extract_generation_text(result) for result in batch_results)
        print(f"{model_name} finished candidates {end_idx}/{total}", flush=True)
    definitions_df.loc[todo_df.index, f"{model_name}_definition"] = definitions
    definitions_df.to_csv(output_path, index=False)
    return definitions_df


def define_candidates(definitions_df, model_ids, gpu_ids, batch_size, output_path):
    check_cuda_gpus(required_gpus=max(gpu_ids) + 1)
    definitions_df = definitions_df.copy()
    for model_name, model_id, gpu_id in zip(MODEL_NAMES, model_ids, gpu_ids):
        generator = None
        try:
            generator = load_model(model_id, gpu_id)
            definitions_df = define_candidates_for_model(
                definitions_df,
                model_name,
                generator,
                batch_size,
                output_path,
            )
        finally:
            print(f"Unloading {model_name} model from GPU...", flush=True)
            unload_model(generator)
    return definitions_df


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mt_input = args.mt_input or args.output_dir.parent / "mt_outputs" / "mt_preds_all.csv"
    definitions_output = args.definitions_output or args.output_dir / "_ma_definitions_tmp.csv"
    dataset_df = load_normalized_dataset(args.dataset, args.limit_rows)
    definitions_df = load_mt_candidates(mt_input, dataset_df)
    definitions_df = define_candidates(definitions_df, args.model_ids, parse_gpu_ids(args.gpu_ids), args.batch_size, definitions_output)
    definitions_df.to_csv(definitions_output, index=False)
    print(f"Saved temporary definitions: {definitions_output}", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
