# -*- coding: utf-8 -*-

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import CrossEncoder


JUDGE = "cross-encoder/stsb-roberta-large"
PAIRS = [("llama", "qwen"), ("llama", "mistral"), ("qwen", "mistral")]


def parse_args():
    parser = argparse.ArgumentParser(description="Score cross-model agreement and sweep mean-similarity thresholds.")
    parser.add_argument("--definitions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scores-output", type=Path, default=None)
    parser.add_argument("--thresholds-output", type=Path, default=None)
    parser.add_argument("--judge-model", default=JUDGE)
    parser.add_argument("--num-thresholds", type=int, default=25)
    parser.add_argument("--limit-rows", type=int, default=None)
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


def check_cuda_gpus(required_gpus=1):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Request a GPU in the SLURM job.")
    gpu_count = torch.cuda.device_count()
    if gpu_count < required_gpus:
        raise RuntimeError(f"Need {required_gpus} CUDA GPU, but only {gpu_count} are visible.")
    print(f"Visible CUDA GPUs: {gpu_count}", flush=True)


def load_similarity_judge(judge_model):
    check_cuda_gpus(required_gpus=1)
    print(f"Loading similarity judge: {judge_model}", flush=True)
    return CrossEncoder(judge_model, device="cuda:0")


def score_agreement(definitions_df, judge):
    scored_df = definitions_df.copy()
    score_mask = scored_df["mt_pred"].apply(to_binary).eq(1)
    for model_a, model_b in PAIRS:
        col_a = f"{model_a}_definition"
        col_b = f"{model_b}_definition"
        score_col = f"{model_a}_{model_b}_similarity"
        scored_df[score_col] = np.nan
        pairs = list(zip(scored_df.loc[score_mask, col_a].fillna("").astype(str), scored_df.loc[score_mask, col_b].fillna("").astype(str)))
        print(f"Scoring {score_col} for {len(pairs)} MT-positive candidates...", flush=True)
        if pairs:
            scored_df.loc[score_mask, score_col] = judge.predict(pairs)

    score_cols = [f"{a}_{b}_similarity" for a, b in PAIRS]
    scored_df["mean_similarity"] = scored_df[score_cols].mean(axis=1)
    scored_df["min_similarity"] = scored_df[score_cols].min(axis=1)
    scored_df["std_similarity"] = scored_df[score_cols].std(axis=1)
    scored_df["ma_score"] = 1.0 - scored_df["mean_similarity"]
    return scored_df


def build_thresholds(scores, num_thresholds):
    scores = np.asarray(scores, dtype=float)
    scores = scores[np.isfinite(scores)]
    if len(scores) == 0:
        return np.array([])
    unique_scores = np.unique(scores)
    if len(unique_scores) <= num_thresholds:
        return np.round(unique_scores, 6)
    return np.round(np.unique(np.quantile(scores, np.linspace(0, 1, num_thresholds))), 6)


def calculate_counts(y_true, y_pred):
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    return tp, fp, fn, tn


def calculate_metrics(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def evaluate_thresholds(scored_df, num_thresholds):
    eval_df = scored_df[scored_df["mean_similarity"].notna()].copy()
    y_true = eval_df["is_gold"].astype(int).values
    scores = eval_df["mean_similarity"].astype(float).values
    mt_mask = eval_df["mt_pred"].apply(to_binary).eq(1).values
    rows = []
    for threshold in build_thresholds(scores, num_thresholds):
        y_pred = (mt_mask & (scores <= threshold)).astype(int)
        tp, fp, fn, tn = calculate_counts(y_true, y_pred)
        precision, recall, f1 = calculate_metrics(tp, fp, fn)
        rows.append(
            {
                "score_category": "mean_similarity",
                "score_column": "mean_similarity",
                "prediction_rule": "mean_similarity <= threshold",
                "threshold": threshold,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "predicted_count": int(y_pred.sum()),
                "gold_count": int(y_true.sum()),
                "candidate_count": int(len(y_true)),
                "is_best": 0,
                "selected_for_prediction": 0,
            }
        )
    threshold_df = pd.DataFrame(rows)
    if not threshold_df.empty:
        best_idx = threshold_df.sort_values(["f1", "precision", "recall", "threshold"], ascending=[False, False, False, True]).index[0]
        threshold_df.loc[best_idx, ["is_best", "selected_for_prediction"]] = 1
    return threshold_df


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scores_output = args.scores_output or args.output_dir / "ma_scoring_breakdown.csv"
    thresholds_output = args.thresholds_output or args.output_dir / "ma_thr_metrics.csv"

    definitions_df = pd.read_csv(args.definitions)
    if args.limit_rows is not None:
        definitions_df = definitions_df.head(args.limit_rows).copy()
    required = ["sentence", "candidate", "cand_type", "is_gold", "mt_pred", "llama_definition", "qwen_definition", "mistral_definition"]
    require_columns(definitions_df, required, args.definitions)

    judge = load_similarity_judge(args.judge_model)
    scored_df = score_agreement(definitions_df, judge)
    scored_df.to_csv(scores_output, index=False)
    print(f"Saved scored agreement results: {scores_output}", flush=True)

    threshold_df = evaluate_thresholds(scored_df, args.num_thresholds)
    threshold_df.to_csv(thresholds_output, index=False)
    print(f"Wrote threshold metrics to {thresholds_output}", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
