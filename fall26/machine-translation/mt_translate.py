"""For a given text input, prompt an LLM to perform 4 separate round trip translations.
The input contains only sentence_id and sentence."""

# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import pandas as pd
import regex as re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline

REQUIRED_COLUMNS = ["sentence_id", "sentence"]

# KEEP THIS ACROSS TESTINGS TO ENSURE CONSISTENT TOKENIZATION
# Words with hyphens or apostrophes are treated as one word. 
WORD_RE = re.compile(
    r"[\p{L}\p{M}\p{N}]+(?:['’\-][\p{L}\p{M}\p{N}]+)*"
)

INPUT_LANGUAGE = {
  "en": "English",
  "pl": "Polish",
  "da": "Danish",
}

TRANSLATION_REGISTRY = {
    "spanish": "Spanish",
    "arabic": "Arabic",
    "chinese": "Chinese",
    "japanese": "Japanese",
}

MODEL_REGISTRY = {
    "llama": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "qwen": "Qwen/Qwen2.5-7B-Instruct",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    "gemma": "google/gemma-2-9b-it",
}


# Handles possible output shapes from chat
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


# Single user message for cross model compatibility
def make_chat(text, lang1, lang2):
    return [
        {
            "role": "user",
            "content": (
                "You are a precise translation engine. "
                f"Translate the following {lang1} text into {lang2}. "
                f"Return only the translation. Do not explain. \n\n"
                f"Text: {text}"
            ),
        }
    ]


# Load model with 4-bit quantization
def load_generator(model_id):

    # Check CUDA availability
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is NOT available. Request GPU for this job."
        )
    print(f"CUDA available. Using GPU now.", flush=True)

    print(f"Loading tokenizer: {model_id}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # H100/A100-class GPUs support bfloat16. Float16 keeps the program usable
    # when Slurm assigns an older GPU such as a V100.
    major_capability, _ = torch.cuda.get_device_capability(0)
    compute_dtype = torch.bfloat16 if major_capability >= 8 else torch.float16
    print(f"Loading model with compute dtype: {compute_dtype}", flush=True)
    
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quant_config,
        torch_dtype=compute_dtype,
        device_map={"": 0},
    )
    model.generation_config.temperature = None
    model.generation_config.top_p = None

    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        do_sample=False,
        return_full_text=False,
    )


# Helper function to check if required columns exist in input
def require_columns(df, reqs, path):
    missing = []
    for col in reqs:
        if col not in df.columns:
            missing.append(col)
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")


# Return a dictionary of the dataset by sentences
# Do not tokenize since we prompt by the full sentence not the canidate
def load_input_rows(path):
    df = pd.read_csv(path)
    require_columns(df, REQUIRED_COLUMNS, path)

    df = df.dropna(subset=REQUIRED_COLUMNS).copy()  # Remove rows with missing information in required columns
    df["sentence"] = df["sentence"].astype(str) # Convert sentences to Python strings

    # Verify that each sentence_id refers to only one unique sentence.
    sentence_counts = (df.groupby("sentence_id")["sentence"].nunique())
    conflicting_ids = sentence_counts[sentence_counts > 1].index.tolist()
    if conflicting_ids:
        raise ValueError(
            "Some sentence IDs refer to multiple different sentences. "
            f"Conflicting IDs include: {conflicting_ids[:10]}"
        )
    
    # Repeated rows are safe now because identical sentence IDs have the
    # same sentence. Keep only the first row for each sentence_id.
    df = df.drop_duplicates(
        subset=["sentence_id"],
        keep="first"
    )

    return df[REQUIRED_COLUMNS].to_dict("records")
    

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--input-lang", type=str, required=True, choices=sorted(INPUT_LANGUAGE))
    parser.add_argument("--output", type=Path, default=Path("_mt_translations_tmp.csv"))
    
    parser.add_argument(
        "--translations",
        nargs="+",
        choices=sorted(TRANSLATION_REGISTRY),
        required=True,
        metavar="TRANSLATION",
        help="Select any combination of translation languages, ranging from 1 to 4 languages."
    )

    parser.add_argument(
        "--model", 
        choices=sorted(MODEL_REGISTRY),
        required=True,
        metavar="MODEL"
    )

    parser.add_argument("--batch-size", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Load dataset
    input_lang = INPUT_LANGUAGE[args.input_lang]
    rows = load_input_rows(args.dataset)
    print(f"Loaded {len(rows)} input sentences from {args.dataset}", flush=True)
    
    # Load model
    model_id = MODEL_REGISTRY[args.model]
    generator = load_generator(model_id)

    # Run by batch size
    output_rows = []
    for start_index in range(0, len(rows), args.batch_size):
        # Select a batch
        end_index = min(start_index + args.batch_size, len(rows))
        batch = rows[start_index:end_index]
        batch_output_rows = [
            {
                "sentence_id": row["sentence_id"],
                "sentence": row["sentence"]
            }
            for row in batch
        ]

        # For each translation lang by args
        for translation_alias in args.translations:
            translation_lang = TRANSLATION_REGISTRY[translation_alias]

            # For each input sentence in a batch, prompt a FORWARD translation
            batch_results_forward = generator(
                [make_chat(row["sentence"], input_lang, translation_lang) for row in batch],
                batch_size=args.batch_size,
            )
            # Extract the forward translations
            translated_rows_forward = [extract_generation_text(result) for result in batch_results_forward]
           
            # For each input sentence in a batch, prompt a BACKWARD translation
            batch_results_backward = generator(
                [make_chat(row, translation_lang, input_lang) for row in translated_rows_forward],
                batch_size=args.batch_size,
            )
            # Extract the backward translations
            translated_rows_backward = [extract_generation_text(result) for result in batch_results_backward]
    
            # Add this language's translations to each sentence dictionary
            for output_row, forward_text, backward_text in zip(batch_output_rows, translated_rows_forward, translated_rows_backward):
                output_row[f"{translation_alias}_translated_sentence"] = forward_text
                output_row[f"{translation_alias}_round_trip_sentence"] = backward_text

        # Save batch results
        output_rows.extend(batch_output_rows)
        print(f"Translated sentences {end_index}/{len(rows)}", flush=True)

    output_df = pd.DataFrame(output_rows)
    output_df.to_csv(args.output, index=False)
    print(f"Wrote {len(output_df)} round-trip translations to {args.output}", flush=True)


if __name__ == "__main__":
    main()
