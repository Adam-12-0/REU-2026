# Candidate Detection: Cross Model Agreement Scoring

print("Starting scoring of agreement...", flush=True)

# Library imports
import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import regex as re
import torch
from rapidfuzz import fuzz
from datasets import load_dataset
from sklearn.metrics import precision_recall_fscore_support

from sentence_transformers import CrossEncoder

JUDGE = "cross-encoder/stsb-roberta-large"
OUTPUT_DIR = Path("results")

DEFINITIONS_FILE = "definitions.csv"
SCORED_FILE = "scores.csv"

MODEL_NAMES = ["llama", "qwen", "mistral"]
PAIRS = [
    ("llama", "qwen"),
    ("llama", "mistral"),
    ("qwen", "mistral"),
]


# Helper functions to load judge model
def check_cuda_gpus(required_gpus=1):
  if not torch.cuda.is_available():
    raise RuntimeError(
      "CUDA is not available. Request a GPU in the SLURM job before running score.py."
    )

  gpu_count = torch.cuda.device_count()
  if gpu_count < required_gpus:
    raise RuntimeError(
      f"score.py needs {required_gpus} CUDA GPU, but only {gpu_count} are visible."
    )

  print("CUDA available: True", flush=True)
  print(f"Visible CUDA GPUs: {gpu_count}", flush=True)
  for gpu_id in range(gpu_count):
    print(f"GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}", flush=True)



def load_similarity_judge():
  check_cuda_gpus(required_gpus=1)
  device = "cuda:0"
  print(f"Loading similarity judge: {JUDGE}", flush=True)
  judge = CrossEncoder(JUDGE, device=device)
  print("Similarity judge loaded successfully", flush=True)
  return judge



# Score agreement between all possible definition pairs
def score_agreement(definitions_df, judge):
  scored_df = definitions_df.copy()

  # Score each pair
  for model_a, model_b in PAIRS:
    col_a = f"{model_a}_definition"
    col_b = f"{model_b}_definition"
    score_col = f"{model_a}_{model_b}_similarity"

    pairs = list(
      zip(
        scored_df[col_a].fillna("").astype(str),
        scored_df[col_b].fillna("").astype(str),
      )
    )

    print(f"Scoring {score_col}...", flush=True)
    scored_df[score_col] = judge.predict(pairs)

  score_cols = [f"{a}_{b}_similarity" for a, b in PAIRS]
  scored_df["mean_similarity"] = scored_df[score_cols].mean(axis=1)
  scored_df["min_similarity"] = scored_df[score_cols].min(axis=1)
  scored_df["std_similarity"] = scored_df[score_cols].std(axis=1)

  output_cols = [
    "sentence",
    "candidate",
    "llama_definition",
    "qwen_definition",
    "mistral_definition",
    "llama_qwen_similarity",
    "llama_mistral_similarity",
    "qwen_mistral_similarity",
    "mean_similarity",
    "min_similarity",
    "std_similarity",
  ]

  optional_cols = [
    "doc_id",
    "candidate_norm",
    "is_gold",
  ]

  existing_cols = [col for col in output_cols + optional_cols if col in scored_df.columns]
  return scored_df[existing_cols]



# Read arguments
def parse_args():
    parser = argparse.ArgumentParser(
        description="Score cross-model definition agreement."
    )
    parser.add_argument(
        "--limit-rows",
        type=int,
        default=None,
        help="Optional small dataset slice for testing.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(OUTPUT_DIR),
    )
    return parser.parse_args()



def main():
  args = parse_args()
  output_dir = Path(args.output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)

  # Ensure definitions are available
  definitions_path = output_dir / DEFINITIONS_FILE
  if not definitions_path.exists():
    raise FileNotFoundError(f"Missing definitions file: {definitions_path}")

  print(f"Loading definitions: {definitions_path}", flush=True)
  definitions_df = pd.read_csv(definitions_path)

  # Optional small test slice
  if args.limit_rows is not None:
    definitions_df = definitions_df.head(args.limit_rows).copy()

  required_cols = {
    "sentence",
    "candidate",
    "llama_definition",
    "qwen_definition",
    "mistral_definition",
  }
  missing = required_cols - set(definitions_df.columns)
  if missing:
    raise ValueError(f"Missing required definition columns: {missing}")

  # Load judge model
  print("Loading agreement judge onto GPU...", flush=True)
  judge = load_similarity_judge()

  print("Scoring agreement...", flush=True)
  scored_df = score_agreement(definitions_df, judge)

  scored_path = output_dir / SCORED_FILE
  scored_df.to_csv(scored_path, index=False)
  print(f"Saved scored agreement results: {scored_path}", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
