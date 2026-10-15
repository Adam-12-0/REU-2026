"""Apply the selected Model Agreement threshold to every word occurrence."""

import argparse
from pathlib import Path

import pandas as pd


# Select the best threshold
def load_selected_threshold(path):
    thresholds = pd.read_csv(path)
    require_columns(thresholds, ["threshold", "selected_for_prediction"], path)
    
    selected = thresholds[thresholds["selected_for_prediction"].astype(int).eq(1)]
    return float(selected.iloc[0]["threshold"])


# Helper function to check if required columns exist in input
def require_columns(df, reqs, path):
    missing = []
    for col in reqs:
        if col not in df.columns:
            missing.append(col)

    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--scores", type=Path, default=Path("_ma_word_scores_tmp.csv"))
    parser.add_argument("--thresholds", type=Path, default=Path("ma_thr_metrics.csv"))
    parser.add_argument("--output", type=Path, default=Path("ma_preds_sw.csv"))
    
    return parser.parse_args()


def main():
    args = parse_args()

    df = pd.read_csv(args.scores)

    require_columns(df, ["mean_similarity", "disagreement_score"], args.scores)
    threshold = load_selected_threshold(args.thresholds)
    
    # Make specialized prediction based on disagreement
    df["ma_pred"] = (df["disagreement_score"].astype(float) >= threshold).astype(int)
    
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Selected minimum disagreement score: {threshold}", flush=True)
    print(f"Wrote {len(df)} word predictions to {args.output}", flush=True)


if __name__ == "__main__":
    main()
