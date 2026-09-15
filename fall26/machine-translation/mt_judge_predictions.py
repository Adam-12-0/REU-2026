# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Apply the selected MT judge threshold to word candidate scores."
    )
    parser.add_argument("--scores", type=Path, default=Path("_mt_word_scores_tmp.csv"))
    parser.add_argument("--output", type=Path, default=Path("mt_preds_sw.csv"))
    parser.add_argument("--thresholds", type=Path, default=Path("thr_metrics.csv"))
    return parser.parse_args()


def require_columns(df, columns, path):
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
       
        
def load_selected_threshold(path):
    thresholds = pd.read_csv(path)

    require_columns(
        thresholds,
        ["threshold", "selected_for_prediction"],
        path,
    )

    selected = thresholds[
        thresholds["selected_for_prediction"].astype(int).eq(1)
    ]

    if selected.empty:
        selected = thresholds.sort_values(
            ["f1", "precision", "recall", "threshold"],
            ascending=[False, False, False, False],
            kind="mergesort",
        ).head(1)

    if selected.empty:
        raise ValueError(f"No selected threshold found in {path}")

    return float(selected.iloc[0]["threshold"])


def main():
    args = parse_args()
    scores_df = pd.read_csv(args.scores)
    if "candidate_type" in scores_df.columns and "cand_type" not in scores_df.columns:
        scores_df = scores_df.rename(columns={"candidate_type": "cand_type"})
    require_columns(
        scores_df,
        ["sentence_id", "sentence", "candidate", "cand_norm", "cand_type", "is_gold", "mt_score"],
        args.scores,
    )
    threshold = load_selected_threshold(args.thresholds)

    output_df = scores_df.copy()
    output_df["mt_pred"] = (output_df["mt_score"].astype(float) >= threshold).astype(int)

    preferred_columns = [
        "sentence_id",
        "sentence",
        "candidate",
        "cand_norm",
        "cand_type",
        "is_gold",
        "token_pos",
        "mt_score",
        "mt_pred",
    ]
    columns = [column for column in preferred_columns if column in output_df.columns]
    for column in output_df.columns:
        if column not in columns:
            columns.append(column)

    output_df[columns].to_csv(args.output, index=False)
    print(f"Selected threshold: {threshold}", flush=True)
    print(f"Wrote {len(output_df)} rows to {args.output}", flush=True)


if __name__ == "__main__":
    main()
