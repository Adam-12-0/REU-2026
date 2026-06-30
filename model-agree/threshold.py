# Candidate Detection: Cross Model Agreement Scores Thresholding


print("Starting threshold analysis...", flush=True)

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support


OUTPUT_DIR = Path("results")
SCORES_FILE = "scores.csv"

SCORE_CONFIGS = [
    ("mean_similarity", "low"),
    ("min_similarity", "low"),
    ("std_similarity", "high"),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Find best agreement thresholds for mean, min, and std similarity."
    )
    parser.add_argument(
        "--scores-path",
        default=None,
        help="Optional path to scores.csv. Defaults to ./scores.csv, then results/scores.csv.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(OUTPUT_DIR),
        help="Directory for threshold plots and metrics CSV.",
    )
    return parser.parse_args()


def find_scores_path(scores_path):
    if scores_path is not None:
        path = Path(scores_path)
        if not path.exists():
            raise FileNotFoundError(f"Missing scores file: {path}")
        return path

    local_path = Path(SCORES_FILE)
    results_path = OUTPUT_DIR / SCORES_FILE

    if local_path.exists():
        return local_path
    if results_path.exists():
        return results_path

    raise FileNotFoundError(
        f"Could not find {SCORES_FILE} in current directory or {OUTPUT_DIR}."
    )


def normalize_gold_labels(labels):
    if labels.dtype == bool:
        return labels.astype(int)

    return (
        labels.astype(str)
        .str.strip()
        .str.lower()
        .map({"1": 1, "true": 1, "yes": 1, "0": 0, "false": 0, "no": 0})
        .fillna(labels)
        .astype(int)
    )


def predict_at_threshold(scores, threshold, direction):
    if direction == "low":
        return (scores <= threshold).astype(int)
    if direction == "high":
        return (scores >= threshold).astype(int)
    raise ValueError(f"Unknown threshold direction: {direction}")


def evaluate_thresholds(df, score_col, direction):
    y_true = normalize_gold_labels(df["is_gold"])
    scores = pd.to_numeric(df[score_col], errors="coerce")

    valid_mask = scores.notna()
    y_true = y_true[valid_mask]
    scores = scores[valid_mask]

    thresholds = np.sort(scores.unique())
    rows = []

    for threshold in thresholds:
        y_pred = predict_at_threshold(scores, threshold, direction)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            average="binary",
            zero_division=0,
        )

        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())

        rows.append(
            {
                "score_column": score_col,
                "direction": direction,
                "threshold": threshold,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "tp": tp,
                "fp": fp,
                "fn": fn,
            }
        )

    eval_df = pd.DataFrame(rows)
    best_idx = eval_df.sort_values(
        ["f1", "precision", "recall"],
        ascending=[False, False, False],
    ).index[0]

    return eval_df, eval_df.loc[best_idx].copy()


def plot_threshold_curve(eval_df, best_row, output_dir):
    score_col = best_row["score_column"]
    plot_path = output_dir / f"{score_col}_threshold_curve.png"

    plt.figure(figsize=(8, 5))
    plt.plot(eval_df["threshold"], eval_df["precision"], label="Precision")
    plt.plot(eval_df["threshold"], eval_df["recall"], label="Recall")
    plt.plot(eval_df["threshold"], eval_df["f1"], label="F1")
    plt.axvline(
        best_row["threshold"],
        linestyle="--",
        color="black",
        label=f"Best threshold = {best_row['threshold']:.3f}",
    )
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title(f"Threshold performance for {score_col}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()

    return plot_path


def plot_counts(best_row, output_dir):
    score_col = best_row["score_column"]
    plot_path = output_dir / f"{score_col}_best_threshold_counts.png"

    counts = [best_row["tp"], best_row["fp"], best_row["fn"]]
    labels = ["TP", "FP", "FN"]

    plt.figure(figsize=(6, 5))
    plt.bar(labels, counts)
    plt.ylabel("Count")
    plt.title(
        f"{score_col} counts at threshold {best_row['threshold']:.3f}"
    )
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()

    return plot_path


def add_predictions(df, best_rows):
    predicted_df = df.copy()

    for best_row in best_rows:
        score_col = best_row["score_column"]
        direction = best_row["direction"]
        threshold = best_row["threshold"]
        prediction_col = score_col.replace("_similarity", "_predicted_is_gold")

        scores = pd.to_numeric(predicted_df[score_col], errors="coerce")
        predicted_df[prediction_col] = predict_at_threshold(
            scores,
            threshold,
            direction,
        )

    return predicted_df
    

def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scores_path = find_scores_path(args.scores_path)
    print(f"Loading scores: {scores_path}", flush=True)
    df = pd.read_csv(scores_path)

    required_cols = {"is_gold"} | {score_col for score_col, _ in SCORE_CONFIGS}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    best_rows = []

    for score_col, direction in SCORE_CONFIGS:
        print(f"Evaluating thresholds for {score_col}...", flush=True)
        eval_df, best_row = evaluate_thresholds(df, score_col, direction)

        curve_path = plot_threshold_curve(eval_df, best_row, output_dir)
        counts_path = plot_counts(best_row, output_dir)

        best_row["threshold_plot"] = str(curve_path)
        best_row["counts_plot"] = str(counts_path)
        best_rows.append(best_row)

        print(
            f"Best {score_col} threshold: {best_row['threshold']:.3f} "
            f"(precision={best_row['precision']:.3f}, "
            f"recall={best_row['recall']:.3f}, f1={best_row['f1']:.3f})",
            flush=True,
        )

    metrics_df = pd.DataFrame(best_rows)
    metrics_path = output_dir / "threshold_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False, float_format="%.3f")
    print(f"Saved threshold metrics: {metrics_path}", flush=True)
    
    predicted_df = add_predictions(df, best_rows)
    predicted_path = output_dir / "scores_with_predictions.csv"
    predicted_df.to_csv(predicted_path, index=False, float_format="%.3f")
    print(f"Saved scores with predicted labels: {predicted_path}", flush=True)



if __name__ == "__main__":
    main()
