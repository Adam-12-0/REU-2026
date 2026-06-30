# Candidate Detection: Cross Model Agreement

print ("Starting cross model agreement...", flush=True)

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

# Model imports
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

print("Imports complete", flush=True)

MODEL1 = "meta-llama/Meta-Llama-3.1-8B-Instruct"
MODEL2 = "Qwen/Qwen2.5-7B-Instruct"
MODEL3 = "mistralai/Mistral-7B-Instruct-v0.3"
DATASET_ID = "acader/genz-alpha-slangs"
OUTPUT_DIR = Path("results")

DEFINITION_MODELS = [
    ("llama", MODEL1, 0),
    ("qwen", MODEL2, 1),
    ("mistral", MODEL3, 1)
]

STOPWORDS = {
    "a", "an", "the",
    "and", "or", "but",
    "to", "of", "in", "on", "at", "by", "for", "from", "with", "about", "as",
    "is", "are", "was", "were", "be", "been", "being",
    "am", "do", "does", "did", "have", "has", "had",
    "i", "me", "my", "mine",
    "you", "your", "yours",
    "he", "him", "his",
    "she", "her", "hers",
    "it", "its",
    "we", "us", "our", "ours",
    "they", "them", "their", "theirs",
    "this", "that", "these", "those",
}

WORD_RE = re.compile(
    r"[\p{L}\p{M}]+(?:['\u2019-][\p{L}\p{M}]+)?|[\p{N}]+"
)
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
    return [match.group(0) for match in WORD_RE.finditer(str(text))]



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



# Helper functions to load models
def load_definition_pipelines():
  required_gpus = max(gpu_id for _, _, gpu_id in DEFINITION_MODELS) + 1
  check_cuda_gpus(required_gpus=required_gpus)
  generators = {}
  
  for model_name, model_id, gpu_id in DEFINITION_MODELS:
    generators[model_name] = load_model(model_id, gpu_id)
    
  return generators



def check_cuda_gpus(required_gpus=1):

  # Check that all necessary GPU available
  if not torch.cuda.is_available():
    raise RuntimeError(
      "CUDA is not available. Request a GPU in SLURM job befor running script."
    )
    
  gpu_count = torch.cuda.device_count()
  if gpu_count < required_gpus:
    raise RuntimeError(
      f"Need {required_gpus} CUDA GPUS, but only {gpu_count} are visible. Update SLURM GPU request before running script."
    )
    
  print("CUDA available: True", flush=True)
  print(f"Visible CUDA GPUs: {gpu_count}", flush=True)
  for gpu_id in range(gpu_count):
    print(f"GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}", flush=True)



def load_model(model_id, gpu_id):
  print(f"Loading tokenizer: {model_id}", flush=True)
  tokenizer = AutoTokenizer.from_pretrained(model_id)
  tokenizer.padding_side = "left"
  if tokenizer.pad_token is None:
      tokenizer.pad_token = tokenizer.eos_token

  dtype = torch.bfloat16

  print(f"Loading model: {model_id}", flush=True)
  model = AutoModelForCausalLM.from_pretrained(
      model_id,
      torch_dtype=dtype,
      device_map={"": gpu_id},
  )

  model.generation_config.temperature = None
  model.generation_config.top_p = None

  generator = pipeline(
      "text-generation",
      model=model,
      tokenizer=tokenizer,
      max_new_tokens=40,
      do_sample=False,
      return_full_text=False,
  )

  print(f"Model loaded successfully on GPU {gpu_id}: {model_id}", flush=True)
  return generator
    
    

# Helper functions to define candidates
def unique_sentence_candidates(sentence):

  # Don't define repeated words
  seen = set()
  candidates = []
  
  for token_idx, word in enumerate(tokenize_words(sentence)):
    cand_norm = normalize_word(word)
    
    if not cand_norm:
      continue
      
    if cand_norm in STOPWORDS:
      continue
    
    if cand_norm in seen:
      continue
      
    seen.add(cand_norm)
    candidates.append(
      {
        "candidate": word,
        "candidate_norm": cand_norm,
        "token_index": token_idx
      }
    )
    
  return candidates
  
  
  
# Prompt for model
def make_chat(candidate, sentence):
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
      )
    },
    {
      "role": "user",
      "content": (
        f"Sentence: {sentence}\n"
        f"Target term: {candidate}\n"
        "Context-specific definition:"
      )
    }
  ]
  


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

  

def build_candidate_rows(data):
  definition_rows = []
  
  # For each sentence, collect each candidate word
  for row_idx, row in data.iterrows():
    doc_id = row["doc_id"]
    sentence = row["Sentence"]
    cands = unique_sentence_candidates(sentence)
    
    # Progress update
    print(
      f"Defining candidates for row {row_idx + 1}/{len(data)} "
      f"({len(cands)} candidates)",
      flush=True,
    )
    
    for candidate_info in cands:
      candidate = candidate_info["candidate"]
      candidate_norm = candidate_info["candidate_norm"]
      cand_row = {
        "doc_id": doc_id,
        "sentence": sentence,
        "candidate": candidate,
        "candidate_norm": candidate_norm,
        # "token_index": candidate_info["token_index"],
        "is_gold": is_gold_candidate(candidate_norm, row["gold_norms"]),
      }
      
      definition_rows.append(cand_row)
    
  return pd.DataFrame(definition_rows)



def define_candidates(data, generators, batch_size, output_path=None):
  definitions_df = build_candidate_rows(data)

  if definitions_df.empty:
    return definitions_df

  # Run each model over all candidates in batches
  for model_name, generator in generators.items():
    total_candidates = len(definitions_df)
    print(
      f"Generating {model_name} definitions for {total_candidates} candidates...",
      flush=True,
    )

    chats = [
      make_chat(row["candidate"], row["sentence"])
      for _, row in definitions_df.iterrows()
    ]
    definitions = []

    for start_idx in range(0, total_candidates, batch_size):
      end_idx = min(start_idx + batch_size, total_candidates)
      batch_chats = chats[start_idx:end_idx]
      batch_results = generator(batch_chats, batch_size=batch_size)
      definitions.extend(
        extract_generation_text(result) for result in batch_results
      )

      print(
        f"{model_name} finished candidates {end_idx}/{total_candidates}",
        flush=True,
      )

    definitions_df[f"{model_name}_definition"] = definitions

    if output_path is not None:
      definitions_df.to_csv(output_path, index=False)
      print(
        f"Saved partial definitions after {model_name}: {output_path}",
        flush=True,
      )

  return definitions_df
      


# Read arguments
def parse_args():
    parser = argparse.ArgumentParser(
        description="Detect slang candidates using cross-model definition agreement."
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
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for definition generation.",
    )
    return parser.parse_args()
    

# MAIN function
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

    # Load definition models
    print("Loading definition models onto GPU...", flush=True)
    def_generators = load_definition_pipelines()
    
    # Define candidates
    print("Defining candidates...", flush=True)
    definitions_path = output_dir / "definitions.csv"
    definitions_df = define_candidates(
        df,
        def_generators,
        args.batch_size,
        output_path=definitions_path,
    )
    definitions_df.to_csv(definitions_path, index=False)
    print(f"Saved candidate definitions: {definitions_path}", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
