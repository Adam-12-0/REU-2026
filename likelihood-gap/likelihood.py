# Candidate Detection: Likelihood Gap

from transformers import AutoTokenizer, AutoModelForMaskedLM
from datasets import load_dataset
from rapidfuzz import fuzz
from sklearn.metrics import precision_recall_fscore_support
from IPython.display import display
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import math
import regex as re
import emoji


def main():

  # Load tokenizer
  tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

  # Use GPU - expected runtime should be ~10 min
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print("Device:", device)
  if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

  # Load model
  model = AutoModelForMaskedLM.from_pretrained("xlm-roberta-base").to(device)
  model.eval()

  # Dataset
  data = load_dataset("acader/genz-alpha-slangs")
  df = pd.DataFrame(data["train"])

  required_cols = {"Sentence", "Translation", "Explanation"}
  missing = required_cols - set(df.columns)
  if missing:
    raise ValueError(f"Missing required columns: {missing}")

  df["gold_terms"] = (
    df["Explanation"]
    .apply(extract_gold_terms)
    .apply(remove_emoji_gold_terms)
  )
  df["gold_norms"] = df["gold_terms"].apply(
    lambda xs: [normalize_term(x) for x in xs if normalize_term(x)]
  )
  df["gold_word_norms"] = df["gold_norms"].apply(
    lambda xs: [x for x in xs if count_words(x) == 1]
  )
  df["gold_phrase_norms"] = df["gold_norms"].apply(
    lambda xs: [x for x in xs if count_words(x) > 1]
  )

  print(df[["Sentence", "gold_terms"]].head(5))

  # Score every word and phrase (4 words) per sentence
  all_candidates_df = detect_dataset_terms(
    df,
    tokenizer,
    model,
    max_phrase_words=4
  )

  # Separate results by words versus phrases
  word_candidates_df = all_candidates_df[
    all_candidates_df["candidate type"] == "word"
  ].copy()

  phrase_candidates_df = all_candidates_df[
    all_candidates_df["candidate type"] == "phrase"
  ].copy()

  all_candidates_df.to_csv(
    "detector_features_all_candidates.csv",
    index=False,
    float_format="%.3f"
  )
  word_candidates_df.to_csv(
    "detector_features_word_candidates.csv",
    index=False,
    float_format="%.3f"
  )
  phrase_candidates_df.to_csv(
    "detector_features_phrase_candidates.csv",
    index=False,
    float_format="%.3f"
  )

  print("\nAll candidates:")
  print(all_candidates_df.sort_values("average gap", ascending=False).head(20))

  print("\nWord candidates:")
  print(word_candidates_df.sort_values("average gap", ascending=False).head(20))

  print("\nPhrase candidates:")
  print(phrase_candidates_df.sort_values("average gap", ascending=False).head(20))

  # Threshold evaluation
  threshold_results_df, best_thresholds_df = evaluate_thresholds_by_candidate_set(
    all_candidates_df,
    word_candidates_df,
    phrase_candidates_df
  )

  threshold_results_df.to_csv(
    "threshold_results_by_candidate_set.csv",
    index=False,
    float_format="%.3f"
  )
  best_thresholds_df.to_csv(
    "best_thresholds_by_candidate_set.csv",
    index=False,
    float_format="%.3f"
  )

  print("\nBest threshold summary:")
  display(best_thresholds_df)

  return (
    all_candidates_df,
    word_candidates_df,
    phrase_candidates_df,
    threshold_results_df,
    best_thresholds_df
  )


# Builds candidates for every sentence in the dataset.
def detect_dataset_terms(df, tokenizer, model, max_phrase_words=4):
  rows = []

  for row_idx, row in df.iterrows():
    if row_idx % 25 == 0:
      print(f"Scoring sentence {row_idx + 1}/{len(df)}") # Progress update

    sentence = row["Sentence"]
    gold_terms = row["gold_terms"]
    gold_norms = row["gold_norms"]
    gold_word_norms = row["gold_word_norms"]
    gold_phrase_norms = row["gold_phrase_norms"]

    # Score all words and phrases in sentence
    sentence_features = detect_terms(
      sentence,
      tokenizer,
      model,
      gold_norms,
      gold_word_norms,
      gold_phrase_norms,
      max_phrase_words=max_phrase_words
    )

    for feature in sentence_features:
      feature["sentence index"] = row_idx
      feature["sentence"] = sentence
      feature["gold terms"] = "; ".join(gold_terms)
      rows.append(feature)

  return pd.DataFrame(rows)


# Extracts gold terms from the Explanation column.
def extract_gold_terms(explanation):
  if not isinstance(explanation, str):
    return []

  # Hard coded extraction according to dataset
  terms = re.findall(r'\*\*"([^"]+)"\*\*:', explanation)

  if not terms:
    terms = re.findall(r'"([^"]+)"\s*:', explanation)

  return [t.strip() for t in terms if t.strip()]


# Detects emojis.
def is_emoji(text):
  return any(ch in emoji.EMOJI_DATA for ch in str(text))


# Removes emojis from gold terms.
def remove_emoji_gold_terms(term_list):
  return [t for t in term_list if not is_emoji(t)]


# Normalize text for candidate/gold matching.
def normalize_term(text):
  text = str(text).lower().strip()
  text = re.sub(r"^[^\p{L}\p{N}]+|[^\p{L}\p{N}]+$", "", text)
  text = re.sub(r"\s+", " ", text)
  return text


# Counts words
def count_words(text):
  text = normalize_term(text)
  if not text:
    return 0
  return len(text.split())


# Returns 1 when a candidate matches one of the extracted gold terms.
def is_gold_candidate(candidate_norm, gold_norms):
  for g in gold_norms:
    if not g:
      continue

    if candidate_norm == g:
      return 1

    # Loose match helps when punctuation/plurals differ slightly.
    if fuzz.ratio(candidate_norm, g) >= 92:
      return 1

  return 0


# Evaluate precision, recall, and F1 for the four detector scores across
# all candidates, word candidates only, and phrase candidates only.
def evaluate_thresholds_by_candidate_set(
  all_candidates_df,
  word_candidates_df,
  phrase_candidates_df,
  num_thresholds=25
):
  candidate_sets = {
    "all candidates": {
      "df": all_candidates_df,
      "gold column": "is_gold",
      "gold comparison": "all gold terms"
    },
    "word candidates": {
      "df": word_candidates_df,
      "gold column": "is_gold_word",
      "gold comparison": "gold words only"
    },
    "phrase candidates": {
      "df": phrase_candidates_df,
      "gold column": "is_gold_phrase",
      "gold comparison": "gold phrases only"
    }
  }

  score_columns = {
    "surprise": "candidate surprise",
    "avg surprise": "candidate avg surprise",
    "gap": "gap",
    "avg gap": "average gap"
  }

  all_results = []
  best_rows = []

  # For the 3 candidate sets
  for candidate_set_name, candidate_set_info in candidate_sets.items():
    cand_df = candidate_set_info["df"]
    gold_col = candidate_set_info["gold column"]
    gold_comparison = candidate_set_info["gold comparison"]

    # Evaluate all 4 scoring methods
    for score_name, score_col in score_columns.items():
      result_df = evaluate_single_threshold_curve(
        cand_df,
        candidate_set_name,
        score_name,
        score_col,
        gold_col,
        gold_comparison,
        num_thresholds=num_thresholds
      )

      if result_df.empty:
        continue

      all_results.append(result_df)

      best_row = result_df.sort_values(
        ["f1", "precision", "recall"],
        ascending=False
      ).iloc[0].to_dict()
      best_rows.append(best_row)

      print(f"\n{candidate_set_name} - {score_name} threshold chart")
      display(result_df)
      plot_threshold_results(result_df, candidate_set_name, score_name)

  threshold_results_df = pd.concat(all_results, ignore_index=True)
  best_thresholds_df = pd.DataFrame(best_rows).sort_values(
    ["candidate set", "f1"],
    ascending=[True, False]
  )

  return threshold_results_df, best_thresholds_df


# Evaluates threshold over 1 score column
def evaluate_single_threshold_curve(
  cand_df,
  candidate_set_name,
  score_name,
  score_col,
  gold_col,
  gold_comparison,
  num_thresholds=25
):
  if cand_df.empty:
    return pd.DataFrame()

  eval_df = cand_df[[gold_col, score_col]].dropna().copy()

  if eval_df.empty:
    return pd.DataFrame()

  y_true = eval_df[gold_col].astype(int).values
  scores = eval_df[score_col].astype(float).values
  thresholds = build_thresholds(scores, num_thresholds=num_thresholds)

  rows = []

  for threshold in thresholds:
    y_pred = (scores >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
      y_true,
      y_pred,
      average="binary",
      zero_division=0
    )

    rows.append({
      "candidate set": candidate_set_name,
      "score category": score_name,
      "score column": score_col,
      "gold column": gold_col,
      "gold comparison": gold_comparison,
      "threshold": threshold,
      "precision": precision,
      "recall": recall,
      "f1": f1,
      "predicted count": int(y_pred.sum()),
      "gold count": int(y_true.sum()),
      "candidate count": int(len(y_true))
    })

  return pd.DataFrame(rows)


# Helper function for threshold evaluation
def build_thresholds(scores, num_thresholds=25):
  scores = np.asarray(scores, dtype=float)
  scores = scores[np.isfinite(scores)]

  if len(scores) == 0:
    return np.array([])

  unique_scores = np.unique(scores)

  if len(unique_scores) <= num_thresholds:
    return np.round(unique_scores, 6)

  quantiles = np.linspace(0, 1, num_thresholds)
  thresholds = np.quantile(scores, quantiles)
  return np.round(np.unique(thresholds), 6)


# Helper function for threshold evaluation
def plot_threshold_results(result_df, candidate_set_name, score_name):
  plt.figure(figsize=(7, 4))
  plt.plot(result_df["threshold"], result_df["precision"], marker="o", label="precision")
  plt.plot(result_df["threshold"], result_df["recall"], marker="o", label="recall")
  plt.plot(result_df["threshold"], result_df["f1"], marker="o", label="f1")
  plt.xlabel(f"{score_name} threshold")
  plt.ylabel("Metric")
  plt.title(f"Method 2: {candidate_set_name.title()} - {score_name.title()} Threshold Results")
  plt.legend()
  plt.grid(True, alpha=0.3)

  plot_dir = Path("threshold_plots")
  plot_dir.mkdir(exist_ok=True)

  filename = f"threshold_plot_{candidate_set_name}_{score_name}.png"
  filename = filename.replace(" ", "_")
  output_path = plot_dir / filename

  plt.tight_layout()
  plt.savefig(output_path, dpi=200, bbox_inches="tight")
  plt.close()

  print(f"Saved plot: {output_path}")


# Detects both individual words and multiword phrase spans.
# Words are spans of length 1. Phrases are spans of length 2..max_phrase_words.
def detect_terms(
  sentence,
  tokenizer,
  model,
  gold_norms,
  gold_word_norms,
  gold_phrase_norms,
  max_phrase_words=4
):

  # Normalize sentence
  sentence = remove_emojis(sentence)
  sentence = clean_candidate_text(sentence)

  # Tokenize sentence
  words = sentence.split()
  features = []

  # Test candidates of 1-4 words
  for span_len in range(1, max_phrase_words + 1):
    for start_idx in range(0, len(words) - span_len + 1):
      end_idx = start_idx + span_len
      score = score_span_in_context(
        words,
        start_idx,
        end_idx,
        tokenizer,
        model,
        gold_norms,
        gold_word_norms,
        gold_phrase_norms
      )

      if score is None:
        continue

      features.append(score)

  return features


""" Main calculation of likelihood gap. Scores a word or phrase by replacing the whole span with one mask per subword."""
def score_span_in_context(
  words,
  start_idx,
  end_idx,
  tokenizer,
  model,
  gold_norms,
  gold_word_norms,
  gold_phrase_norms
):

  # Gets selected candidate
  candidate_words = words[start_idx:end_idx]
  candidate = " ".join(candidate_words)
  candidate_norm = normalize_term(candidate)
  
  # Tokenize candidate into subwords
  cand_ids = tokenizer.encode(candidate, add_special_tokens=False)

  if len(cand_ids) == 0 or not candidate_norm:
    return None

  num_words = end_idx - start_idx
  num_subwords = len(cand_ids)
  cand_tokens = tokenizer.convert_ids_to_tokens(cand_ids)

  # Masks tokens
  mask_span = " ".join([tokenizer.mask_token] * num_subwords)
  masked_words = words[:start_idx] + [mask_span] + words[end_idx:]
  masked_sentence = " ".join(masked_words)

  # Create masked sentence
  inputs = tokenizer(masked_sentence, return_tensors="pt")
  device = next(model.parameters()).device
  inputs = {key: value.to(device) for key, value in inputs.items()}

  mask_positions = (
    inputs["input_ids"] == tokenizer.mask_token_id
  ).nonzero(as_tuple=True)[1]

  if len(mask_positions) != num_subwords:
    raise ValueError(
      f"Expected {num_subwords} masks for {candidate}, found {len(mask_positions)}"
    )

  with torch.no_grad():
    outputs = model(**inputs)

  # Probability distribution
  logits = outputs.logits[0]

  candidate_log_prob = 0.0
  top_log_prob = 0.0
  top_tokens = []
  top_token_probs = []

  # For each masked position, save the probability of the actual candidate, and the probability of the highest likely prediction
  for mask_idx, subword_id in zip(mask_positions, cand_ids):
    log_probs = torch.log_softmax(logits[mask_idx], dim=-1)

    candidate_log_prob += log_probs[subword_id].item()

    top_id = torch.argmax(log_probs).item()
    top_log_prob += log_probs[top_id].item()
    top_tokens.append(tokenizer.decode([top_id]).strip())
    top_token_probs.append(math.exp(log_probs[top_id].item()))
  
  # Surprise - indicates how expected actual candidate was
  candidate_prob = math.exp(candidate_log_prob)
  candidate_surprise = -candidate_log_prob
  candidate_avg_surprise = candidate_surprise / num_subwords

  top_token = " ".join(top_tokens)
  top_token_probability = math.exp(top_log_prob)

  # Likelihood gap - indicates how much the model preferred the other candidate
  gap = top_log_prob - candidate_log_prob
  avg_gap = gap / num_subwords # Normalizes values to avoid misrepresentative results for longer or shorter candidates

  candidate_type = "word" if num_words == 1 else "phrase"

  return {
    "candidate": candidate,
    "candidate norm": candidate_norm,
    "candidate type": candidate_type,
    "is_gold": is_gold_candidate(candidate_norm, gold_norms),
    "is_gold_word": is_gold_candidate(candidate_norm, gold_word_norms),
    "is_gold_phrase": is_gold_candidate(candidate_norm, gold_phrase_norms),
    "masked sentence": masked_sentence,
    "subwords": " ".join(cand_tokens),
    "number of subwords": num_subwords,
    "candidate probability": candidate_prob,
    "candidate surprise": candidate_surprise,
    "candidate avg surprise": candidate_avg_surprise,
    "top token": top_token,
    "top token probability": top_token_probability,
    "gap": gap,
    "average gap": avg_gap,
    "start word index": start_idx,
    "end word index": end_idx - 1,
    "number of words": num_words
  }


# Return sentences without emojis.
def remove_emojis(s):
  return re.sub(r"[\U00010000-\U0010ffff]", "", s)


# Remove punctuation from candidates.
def clean_candidate_text(s):
  """
  Removes punctuation attached to candidate words before tokenization.
  Keeps letters, numbers, apostrophes, and hyphens.
  """
  s = re.sub(r"[^\w\s'\-]", "", s)
  s = re.sub(r"\s+", " ", s).strip()
  return s


# Stores result of pipeline into 5 dataframes
(
  all_candidates_df,
  word_candidates_df,
  phrase_candidates_df,
  threshold_results_df,
  best_thresholds_df
) = main()






