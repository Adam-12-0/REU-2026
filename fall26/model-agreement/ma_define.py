"""Generate three LLM definitions for every word occurrence in each unique input.
The input contains only sentence_id and sentence. For each row of text (sentence), 
the program will tokenize it by words and prompt the LLM to define each word in the 
context of the sentence or full text."""

# -*- coding: utf-8 -*-

import argparse
import gc
import os
import unicodedata
from pathlib import Path

import pandas as pd
import regex as re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline


REQUIRED_COLUMNS = ["sentence_id", "sentence"]

MODEL_REGISTRY = {
    "llama": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "qwen": "Qwen/Qwen2.5-7B-Instruct",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    "gemma": "google/gemma-2-9b-it",
}

# KEEP THIS ACROSS TESTINGS TO ENSURE CONSISTENT TOKENIZATION
# Words with hyphens or apostrophes are treated as one word. 
WORD_RE = re.compile(
    r"[\p{L}\p{M}\p{N}]+(?:['’\-][\p{L}\p{M}\p{N}]+)*"
)


# Extract response text from string-style or chat-style output.
def extract_generation_text(result):
    if isinstance(result, list):
        if not result:
            return ""
        result = result[-1]
    if isinstance(result, dict):
        text = result.get("generated_text", result.get("content", ""))
    else:
        text = str(result)

    # Chat style pipeline may return a list of messages
    if isinstance(text, list):
        if not text:
            return ""
        last = text[-1]
        if isinstance(last, dict):
            return str(last.get("content", "")).strip()
        return str(last).strip()
    return str(text).strip()


# Build prompt to define candidate term in text context
def make_chat(candidate, text):
    return [{
        "role": "user",
        "content": (
            "Define the marked term as it is used in the text. Return only a "
            "short definition using no more than 20 words. The definition must be in English.\n\n"
            f"Text: {text}\n"
            f"Term: {candidate}"
        ),
    }]


# Load one LLM with 4-bit quantization
def load_model(model_id, max_new_tokens):
    # Check CUDA availability
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is NOT available. Request GPU for this job.")
    print(f"CUDA available. Using GPU now.", flush=True)

    # Load model tokenizer
    print(f"Loading {model_id}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # H100/A100-class GPUs support bfloat16. Float16 keeps the program usable
    # when Slurm assigns an older GPU such as a V100.
    major_capability, _ = torch.cuda.get_device_capability(0)
    compute_dtype = torch.bfloat16 if major_capability >= 8 else torch.float16
    print(f"Loading model with compute dtype: {compute_dtype}", flush=True)

    # Use 4 bit quantization 
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        torch_dtype=compute_dtype,
        device_map={"": 0}
    )
    model.generation_config.temperature = None
    model.generation_config.top_p = None

    return pipeline(
        "text-generation",
        model=model,
        tokenizer = tokenizer,
        max_new_tokens = max_new_tokens,
        do_sample = False,
        return_full_text = False
    )


# For a single model, load, prompt, and delete it. Return only a list of string definitions.
def generate_definitions(df, model_name, model_id, batch_size, max_new_tokens):
    # Load model
    model = load_model(model_id, max_new_tokens)

    # Run definitions by batch size
    definitions = []
    for start_idx in range(0, len(df), batch_size):

        # Select batch
        end_idx = min(start_idx + batch_size, len(df))
        batch = df.iloc[start_idx:end_idx]

        # For each input candidate in batch, create a prompt according to its text context and receive response
        batch_results = model(
            [make_chat(row.candidate, row.sentence) for row in batch.itertuples(index = False)],
            batch_size=batch_size
        )

        # Pair each input candidate with corresponding model response
        # Pipeline returns results in same order as prompts
        for result in batch_results:
            definition = extract_generation_text(result)
            definitions.append(definition)
        print(f"{model_name} processed candidates {end_idx}/{len(df)}", flush=True) # Batch update

    # Confirm one definition per candidate
    if len(definitions) != len(df):
        raise RuntimeError(f"{model_name} produced {len(definitions)} definitions for {len(df)} candidates.")

    # Release model to open space for next one
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return definitions


# Helper function to normalize term
def normalize_word(term):
    term = unicodedata.normalize("NFKC", str(term))
    term = re.sub(r"\s+", " ", term)
    return term.strip().casefold()


# Helper function to check if required columns exist in input
def require_columns(df, reqs, path):
    missing = []
    for col in reqs:
        if col not in df.columns:
            missing.append(col)

    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")


# Return a DataFrame of the loaded dataset. Verify that each id is a unique input text and check the dataset does not contain errors
# Create one row per word occurrence
def load_input_rows(path):
    df = pd.read_csv(path)
    require_columns(df, REQUIRED_COLUMNS, path)

    df = df.dropna(subset=REQUIRED_COLUMNS).copy() # Remove rows with missing info
    df["sentence"] = df["sentence"].astype(str) # Convert sentence columns to strings

    # Verify each sentence_id refers to unique sentence
    sentence_counts = (df.groupby("sentence_id")["sentence"].nunique())
    conflicting_ids = sentence_counts[sentence_counts > 1].index.tolist()
    if conflicting_ids:
        raise ValueError(
            "Some sentence IDs refer to multiple different sentences. "
            f"Conflicting IDs include: {conflicting_ids[:10]}"
        )
    
    # Repeated rows are safe now since identical sentence IDs have the same sentence.
    # Keep only the first row for each sentence_id
    df = df.drop_duplicates(subset=["sentence_id"], keep="first")

    # Tokenize the input by words
    candidate_rows = []
    for row in df.itertuples(index=False):
        sentence = row.sentence

        for token_pos, match in enumerate(WORD_RE.finditer(sentence)):
            candidate = match.group(0)
            start, end = match.span() # Character positioning

            candidate_rows.append(
                {
                    "sentence_id": row.sentence_id,
                    "sentence": row.sentence,
                    "candidate": candidate,
                    "cand_norm": normalize_word(candidate),
                    "cand_type": "single_word",
                    "token_pos": token_pos,
                    "start_char": start,
                    "end_char": end
                }
            )

    return pd.DataFrame(candidate_rows)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("_ma_definitions_tmp.csv"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=64) # The maximum number of tokens a model may generate
    
    parser.add_argument(
        "--models",
        nargs="+",
        choices=sorted(MODEL_REGISTRY),
        required=True,
        metavar="MODEL",
        help="Two or three distinct aliases chosen from: llama, qwen, mistral, gemma.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Load dataset
    cands = load_input_rows(args.dataset)
    if cands.empty:
        raise ValueError("No word candidates were found in the selected sentences.")

    # For each model, load and prompt it, then delete model
    for model_name in args.models:

        # Append the model's definitions
        cands[f"{model_name}_definition"] = generate_definitions(
            cands,
            model_name,
            MODEL_REGISTRY[model_name],
            args.batch_size,
            args.max_new_tokens
        )

    # Save output csv of candidates with definitions
    cands.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"Wrote {len(cands)} word occurrences to {args.output}", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
