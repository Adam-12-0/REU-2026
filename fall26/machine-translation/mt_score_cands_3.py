"""Receive the translations and compare the original sentence to the final round trip
sentence. For a given candidate in a sentence, prompt an LLM to list terms that were 
lost or changed meaning in translation."""

# -*- coding: utf-8 -*-

import argparse
import unicodedata
from pathlib import Path

import pandas as pd
import regex as re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline

# KEEP THIS ACROSS TESTINGS TO ENSURE CONSISTENT TOKENIZATION
# Words with hyphens or apostrophes are treated as one word. 
WORD_RE = re.compile(
    r"[\p{L}\p{M}\p{N}]+(?:['’\-][\p{L}\p{M}\p{N}]+)*"
)

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
    normalized_text = normalize_word(text).rstrip(".!:")
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
        term_norm = normalize_word(term)
        none_check = term_norm.rstrip(".!:")

        # Skip term if not normalized, none, or duplicate
        if not term_norm or term_norm in seen or none_check in none_variants:
            continue

        # Save term
        seen.add(term_norm)
        terms.append(term)

    return terms


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
            return str(last.get("content", "")).strip()
        return str(last).strip()
    return str(text).strip()


# Build a prompt for the model to determine if a candidate was preserved in translation or lost/changed meaning
def make_chat(original_sentence, round_trip_sentence):
    return [
        {
            "role": "user",
            "content": (
                "Read Text 1 and return individual terms from it that disappeared or changed in meaning in Text 2.\n\n"
                f"Return only the exact terms copied from Text 1, separated by commas.\n"
                f"Do not explain. If there are none, return NONE.\n"
                f"Text 1: {original_sentence}\n"
                f"Text 2: {round_trip_sentence}\n"
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
        max_new_tokens=128,
        do_sample=False,
        return_full_text=False,
    )


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


# Return a DataFrame of the loaded dataset and create one row per single word
def load_input_rows(path):
  df = pd.read_csv(path)
  require_columns(df, ["sentence_id", "sentence"], path)

  # Remove rows missing required info
  df = df.dropna(subset=["sentence_id", "sentence"]).copy()
  # Normalize
  df["sentence"] = df["sentence"].astype(str)

  # Verify each sentence_id refers to unique sentence
  sentence_counts = (df.groupby("sentence_id")["sentence"].nunique())
  conflicting_ids = sentence_counts[sentence_counts > 1].index.tolist()
  if conflicting_ids:
    raise ValueError (
        "Some sentence IDs refer to multiple different sentences. "
        f"Conflicting IDs include: {conflicting_ids[:10]}"
    )
    
  # Repeated rows are safe now since identical sentence IDs have the same sentence.
  # Keep only the first row for each sentence_id
  df = df.drop_duplicates(subset=["sentence_id"], keep="first")

  # Add new columns for candidate info and preserve original columns
  candidate_rows = []
  for original_row in df.to_dict("records"):
    sentence = original_row["sentence"]

    # Tokenize by single words
    for token_pos, match in enumerate(WORD_RE.finditer(sentence)):
      candidate = match.group(0)
      start, end = match.span()

      candidate_row = original_row.copy()

      candidate_row.update(
        {
          "candidate":candidate,
          "cand_norm": normalize_word(candidate),
          "cand_type": "single_word",
          "token_pos": token_pos,
          "start_char": start,
          "end_char": end
        }
      )
      candidate_rows.append(candidate_row)

  return pd.DataFrame(candidate_rows)


def parse_args():
  parser = argparse.ArgumentParser()

  parser.add_argument("--translations", type=Path, default=Path("_mt_translations_tmp.csv"))
  parser.add_argument("--output", type=Path, default=Path("_mt_candidates_tmp.csv"))
  parser.add_argument("--batch-size", type=int, default=4)

  parser.add_argument(
    "--trans-langs",
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

  return parser.parse_args()


def main():
  args = parse_args()
  args.output.parent.mkdir(parents=True, exist_ok=True)

  # Load dataset 
  cands = load_input_rows(args.translations) # DataFrame
  if cands.empty:
        raise ValueError("No candidates were found in the selected sentences.")
  print(f"Loaded {len(cands)} word candidates from {args.translations}", flush=True)

  # Load model
  model_id = MODEL_REGISTRY[args.model]
  generator = load_generator(model_id)

  # Identify unique sentences
  sentence_groups = [group.copy() for _,group in cands.groupby("sentence_id", sort=False)]

  # Run by batch size based on unique sentences
  output_rows = []
  for start_index in range(0, len(sentence_groups), args.batch_size):
    # Select a batch
    end_index = min(start_index + args.batch_size, len(sentence_groups))
    batch_groups = sentence_groups[start_index:end_index] # List of dataframes

    # Per translation language, determine which candidates in the sentence were lost or changed meaning in translation
    for translation_alias in args.trans_langs:
      # Column labels
      round_trip_col = f"{translation_alias}_round_trip_sentence" # This line exists because we don't hardcode the translation langs 
      prediction_col = f"{translation_alias}_pred"

      # Set default predictions to 0
      for group in batch_groups: 
        group[prediction_col] = 0

      # Build prompts per sentence
      chats = []
      for group in batch_groups:
        first_row = group.iloc[0]
        chats.append(make_chat(first_row["sentence"], first_row[round_trip_col]))

      # Run batch of prompts
      results = generator(chats, batch_size=args.batch_size)
      if len(results) != len(batch_groups):
        raise RuntimeError(f"Expected {len(batch_groups)} results but received {len(results)}.")
      
      # Match the result with the corresponding sentence group
      for sentence_group, res in zip(batch_groups, results): # Unzip batch and results synchronously
        raw_response = extract_generation_text(res)
        predicted_terms = parse_predicted_terms(raw_response)

        # Ensure terms are normalized for matching
        predicted_terms = {normalize_word(t) for t in predicted_terms}
        
        # Update prediction to 1 if term listed by model
        sentence_group[prediction_col] = (sentence_group["cand_norm"].isin(predicted_terms).astype(int))

    # Count the number of specialized votes from the translation languages results
    pred_col_labels = [f"{translation_alias}_pred" for translation_alias in args.trans_langs]
    for group in batch_groups:
        group["vote_total"] = group[pred_col_labels].sum(axis=1)
        group["vote_score"] = group["vote_total"] / len(pred_col_labels)
        output_rows.extend(group.to_dict("records"))
    
    print(f"Scored sentences {end_index}/{len(sentence_groups)}", flush=True)

  output_df = pd.DataFrame(output_rows)
  output_df.to_csv(args.output, index=False)
  print(f"Wrote {len(output_df)} candidate scores to {args.output}", flush=True)


if __name__ == "__main__":
    main()