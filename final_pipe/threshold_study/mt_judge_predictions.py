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
    parser.add_argument("--threshold", type=float, default=0.75)
    return parser.parse_args()


def require_columns(df, columns, path):
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")


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
    threshold = args.threshold

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
