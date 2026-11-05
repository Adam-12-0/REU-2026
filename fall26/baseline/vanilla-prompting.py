# -*- coding: utf-8 -*-

"""Prompt an LLM to extract specialized-language spans from input sentences.

The input contains only sentence_id and sentence. The LLM returns a comma-separated
list of specialized terms. Each unique predicted occurrence is written to its own row.
"""

import argparse
from pathlib import Path
import unicodedata

import pandas as pd
import regex as re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline

DEFAULT_MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
REQUIRED_COLUMNS = ["sentence_id", "sentence"]
OUTPUT_COLUMNS = [
    "sentence_id",
    "sentence",
    "special_term",
    "term_type"
]

# Words with hyphens or apostrophes are treated as one word. 
WORD_RE = re.compile(
    r"[\p{L}\p{M}\p{N}]+(?:['’\-][\p{L}\p{M}\p{N}]+)*"
)


# Determines if a prediction is a single or multi word candidate
def tokenize_words(sentence):
    return [
        {
            "token_id": token_id,
            "text": match.group(0),
            "start": match.start(),
            "end": match.end(),
        }
        for token_id, match in enumerate(WORD_RE.finditer(sentence))
    ]

# Normalize a term for duplicate detection and evaluation
def normalize_term(term):
    # Make capitalization and white space irrelevant
    term = unicodedata.normalize("NFKC", str(term))
    term = re.sub(r"\s+", " ", term)
    return term.strip().casefold()


# Parse a model response into unique term strings. Return a list of strings
def parse_predicted_terms(text):
    text = str(text).strip()
    if not text: return []

    none_variants = {
        "none",
        "no specialized terms",
        "no terms",
        "n/a",
        "null"
    }

    # Handle cases of where response NONE means model found no special terms
    normalized_text = normalize_term(text).rstrip(".!:")
    if normalized_text in none_variants: 
        return []

    # Remove unwanted response prefixes
    text = re.sub(r"^\s*(?:specialized terms?|terms?|answer)\s*:\s*","",text,flags=re.IGNORECASE)

    # Tolerate semicolons, pipes, newlines
    pieces = re.split(r"\s*(?:,|;|\||\n)\s*", text)

    # Begin saving terms
    terms = []
    seen = set() # Save terms that already appeared to avoid duplicates

    for piece in pieces:
        # Remove list markers
        term = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*","", piece).strip()
        
        # Remove quotes
        if (
            len(term) >= 2
            and term[0] in {'"', "'"}
            and term[-1] == term[0]
        ):
            term = term[1:-1].strip()

        # Save term after cleaning
        term = re.sub(r"\s+", " ", term).strip()
        term_norm = normalize_term(term)
        none_check = term_norm.rstrip(".!:")

        # Skip term if not normalized, none, or duplicate
        if not term_norm or term_norm in seen or none_check in none_variants:
            continue

        # Save term
        seen.add(term_norm)
        terms.append(term)

    return terms

# Return GPT OSS final channel, excluding internal reasoning
def extract_gptoss_final(text):
    text = str(text).strip()
    
    # Raw Harmony output for if tokens were preserved
    raw_final_marker = "<|channel|>final<|message|>"
    if raw_final_marker in text:
        text = text.rsplit(raw_final_marker, 1)[-1]
        text = re.split(r"<\|(?:return|end)\|>", text, maxsplit=1)[0]
        return text.strip()
        
    # Output after removal
    decoded_final_marker = "assistantfinal"
    if decoded_final_marker in text:
        return text.rsplit(decoded_final_marker, 1)[-1].strip()
        
    if text.startswith("analysis"): 
        return ""
      
    # Return the usual responses from Llama/Qwen/Mistral/Gemma  
    return text


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
    if isinstance(text, list):
        if not text:
            return ""
        last = text[-1]
        if isinstance(last, dict):
            text = last.get("content","")
        else:
            text = str(last)

    return extract_gptoss_final(text)


# Convert one LLM response into one output row per unique predicted term.
def detect(sentence_id, sentence, generator_result):
    raw_output = extract_generation_text(generator_result)
    predicted_terms = parse_predicted_terms(raw_output)

    rows = []
    # Read the full prediction and build a row per predicted special term
    for special_term in predicted_terms:

        # Save special term temporarily if it exists
        special_term = special_term.strip()
        if not special_term: continue

        # Identify candidate type with tokenization
        token_count = len(tokenize_words(special_term))
        if token_count == 0: continue # The prediction contains no word-like text

        # Save prediction
        rows.append(
            {
                "sentence_id": sentence_id,
                "sentence": sentence,
                "special_term": special_term,
                "term_type": ("single_word" if token_count == 1 else "multiword")
            }
        )
    return rows


# Create prompt for LLM to identify specialized language.
# This prompt is compatible with Llama, Qwen, Mistral, and Gemma models.
def make_chat(sentence):
    return [
        {
            "role": "user",
            "content": (
                "Identify specialized terms, slang, coded terms, technical terms, or domain-specific expressions.\n\n"
                "Return only the exact spans copied from the sentence, separated by commas. "
                "Do not explain. If none are present, return NONE.\n\n"
                f"Sentence: {sentence}"
            )
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
        
    if model_id.startswith("openai/gpt-oss"):
      print("Loading GPT-OSS using its native MXFP4 checkpoint.", flush=True)
      model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map="auto")
      
    else:
      
      # Existing loader for Llama, Qwen, Mistral, and Gemma.
      # H100/A100-class GPUs support bfloat16. Float16 keeps the program usable
      # when Slurm assigns an older GPU such as a V100.
      major_capability, _ = torch.cuda.get_device_capability(0)
      compute_dtype = torch.bfloat16 if major_capability >= 8 else torch.float16
      print(f"Loading model with compute dtype: {compute_dtype}", flush=True)
      
      quant_config = BitsAndBytesConfig(
          load_in_4bit=True,
          bnb_4bit_quant_type="nf4",
          bnb_4bit_use_double_quant=True,
          bnb_4bit_compute_dtype=compute_dtype
      )
      model = AutoModelForCausalLM.from_pretrained(
          model_id,
          quantization_config=quant_config,
          torch_dtype=compute_dtype,
          device_map={"": 0}
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


# Verify that each id is a unique sentence and double check the dataset does not have errors
# Return a dictionary when finished loading dataset safely
def load_input_rows(dataset_path):
    df = pd.read_csv(dataset_path)
    require_columns(df, REQUIRED_COLUMNS, dataset_path)

    
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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Load dataset
    rows = load_input_rows(args.dataset)
    print(f"Loaded {len(rows)} input sentences from {args.dataset}", flush=True)

    # Load model
    generator = load_generator(args.model_id)

    # Run by batch size
    output_rows = []
    for start_index in range(0, len(rows), args.batch_size):
        # Select batch
        end_index = min(start_index + args.batch_size, len(rows))
        batch = rows[start_index:end_index]

        # For each input sentence in batch, create a prompt and receive the response
        batch_results = generator(
            [make_chat(row["sentence"]) for row in batch],
            batch_size=args.batch_size,
        )

        # Pair each input sentence with its corresponding model response
        for row, result in zip(batch, batch_results):
            detected_rows = detect(
                row["sentence_id"],
                row["sentence"],
                result
            )
            output_rows.extend(detected_rows)
  
        print(f"Processed sentences {end_index}/{len(rows)}", flush=True) # Batch update

    # Save output of specialized predictions from model prompting
    output_df = pd.DataFrame(output_rows, columns=OUTPUT_COLUMNS)
    output_df.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"Wrote {len(output_df)} predicted terms to {args.output}", flush=True)


if __name__ == "__main__":
    main()
