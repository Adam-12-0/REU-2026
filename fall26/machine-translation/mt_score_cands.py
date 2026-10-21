"""Receive the translations and compare the original sentence to the final round trip
sentence. For a given sentence, check how much it changes or if it's lost by referrig
to rapidfuzz for character based similarity. This only check how different the word changed 
in spelling, not in meaning."""

# -*- coding: utf-8 -*-

import argparse
import unicodedata
from pathlib import Path

import pandas as pd
import regex as re
from rapidfuzz import fuzz

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

# Find the highest term similarity score between the candidate and its round trip sentence
def best_fuzzy_score(cand_norm, round_trip_sentence):
  # Save variable of best similarity
  best_score = 0.0

  # For each word in round trip sentence, calculate similarity
  for match in WORD_RE.finditer(round_trip_sentence):
    current_round_trip_word = match.group(0)
    current_rtw_norm = normalize_word(current_round_trip_word)
    current_score = fuzz.ratio(cand_norm, current_rtw_norm)

    # If current similarity is greater, update it
    best_score = max(best_score, current_score)

  return best_score


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


# Return a DataFrame of the loaded dataset and create one row per word
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
  parser.add_argument("--fuzzy-threshold", type=int, default=93)
  parser.add_argument("--batch-size", type=int, default=4)

  parser.add_argument(
          "--trans-langs",
          nargs="+",
          choices=sorted(TRANSLATION_REGISTRY),
          required=True,
          metavar="TRANSLATION",
          help="Select any combination of translation languages, ranging from 1 to 4 languages."
      )
  return parser.parse_args()


def main():
  args = parse_args()
  args.output.parent.mkdir(parents=True, exist_ok=True)

  # Load dataset by single word candidates
  cands = load_input_rows(args.translations) # DataFrame
  if cands.empty:
        raise ValueError("No word candidates were found in the selected sentences.")
  print(f"Loaded {len(cands)} word candidates from {args.translations}", flush=True)

  # Run by batch size
  output_rows = []
  for start_index in range(0, len(cands), args.batch_size):
    # Select a batch
    end_index = min(start_index + args.batch_size, len(cands))
    batch = cands.iloc[start_index:end_index].copy()

    # Calculate one fuzzy score and one vote per translation language
    for translation_alias in args.trans_langs:
      # Column labels
      round_trip_col = f"{translation_alias}_round_trip_sentence"
      fuzzy_score_col = f"{translation_alias}_fuzzy_score"
      prediction_col = f"{translation_alias}_pred"

      # Save score and prediction
      batch[fuzzy_score_col] = batch.apply(lambda row: best_fuzzy_score(row["cand_norm"], row[round_trip_col]), axis=1)
      batch[prediction_col] = (batch[fuzzy_score_col] < args.fuzzy_threshold).astype(int)

    # Count the number of specialized votes from the translation languages results
    pred_col_labels = [f"{translation_alias}_pred" for translation_alias in args.trans_langs]
    batch["vote_total"] = batch[pred_col_labels].sum(axis=1)
    batch["vote_score"] = batch["vote_total"] / len(pred_col_labels)
    output_rows.extend(batch.to_dict("records"))
    print(f"Scored candidates {end_index}/{len(cands)}", flush=True)

  output_df = pd.DataFrame(output_rows)
  output_df.to_csv(args.output, index=False)
  print(f"Wrote {len(output_df)} candidate scores to {args.output}", flush=True)


if __name__ == "__main__":
    main()