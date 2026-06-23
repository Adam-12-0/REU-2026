# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_INPUT_CSV = Path("detect3_outputs") / "detect3_all_word_scores.csv"
DEFAULT_OUTPUT_DIR = Path("detect3_outputs") / "threshold_analysis"


def as_int_series(series):
    return pd.to_numeric(series, errors="coerce").fillna(0).astype(int)


def require_columns(df, required_cols):
    missing = required_cols - set(df.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"Input CSV is missing required columns: {missing_text}")


def predictions_for_threshold(df, threshold):
    fuzz_flag = pd.to_numeric(df["fuzz_ratio"], errors="coerce").fillna(0) < threshold
    untranslated_flag = (
        (as_int_series(df["term_failed_to_translate"]) == 1)
        & (as_int_series(df["is_named_entity"]) == 0)
    )
    return (fuzz_flag | untranslated_flag).astype(int)


def metrics_for_predictions(y_true, y_pred):
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "predicted_count": tp + fp,
        "gold_count": tp + fn,
        "candidate_count": tp + fp + fn + tn,
    }


def evaluate_thresholds(df, start, stop, step):
    y_true = as_int_series(df["is_gold"])
    rows = []

    for threshold in range(start, stop + 1, step):
        y_pred = predictions_for_threshold(df, threshold)
        row = {"threshold": threshold}
        row.update(metrics_for_predictions(y_true, y_pred))
        rows.append(row)

    return pd.DataFrame(rows)


def choose_best_threshold(thresholds_df):
    # Prefer maximum F1. Break exact ties with higher precision, then higher recall,
    # then the lower threshold to avoid expanding the candidate list unnecessarily.
    ranked = thresholds_df.sort_values(
        ["f1", "precision", "recall", "threshold"],
        ascending=[False, False, False, True],
    )
    return ranked.iloc[0]


def plot_thresholds(thresholds_df, best_row, output_path):
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds_df["threshold"], thresholds_df["precision"], label="precision")
    plt.plot(thresholds_df["threshold"], thresholds_df["recall"], label="recall")
    plt.plot(thresholds_df["threshold"], thresholds_df["f1"], label="f1", linewidth=2.5)

    best_threshold = int(best_row["threshold"])
    best_f1 = float(best_row["f1"])
    plt.axvline(best_threshold, color="black", linestyle="--", alpha=0.55)
    plt.scatter([best_threshold], [best_f1], color="black", zorder=3)
    plt.text(
        best_threshold,
        min(best_f1 + 0.04, 0.96),
        f"best={best_threshold}, f1={best_f1:.3f}",
        ha="center",
        va="bottom",
    )

    plt.ylim(0, 1)
    plt.xlabel("Fuzzy threshold")
    plt.ylabel("Metric")
    plt.title("Method 3: Fuzzy Score Thresholds")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Find and plot the best fuzzy threshold for detect3.py using the "
            "current detect3 decision rule."
        )
    )
    parser.add_argument("--input-csv", default=str(DEFAULT_INPUT_CSV))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--start", type=int, default=0, help="First threshold to test.")
    parser.add_argument("--stop", type=int, default=100, help="Last threshold to test.")
    parser.add_argument("--step", type=int, default=1, help="Threshold step size.")
    return parser.parse_args()


def main():
    args = parse_args()
    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.step <= 0:
        raise ValueError("--step must be greater than 0")
    if args.start > args.stop:
        raise ValueError("--start must be less than or equal to --stop")
    if not input_csv.exists():
        raise FileNotFoundError(
            f"Could not find {input_csv}. Run detect3.py first, or pass --input-csv."
        )

    df = pd.read_csv(input_csv)
    require_columns(
        df,
        {
            "fuzz_ratio",
            "term_failed_to_translate",
            "is_named_entity",
            "is_gold",
        },
    )

    thresholds_df = evaluate_thresholds(df, args.start, args.stop, args.step)
    best_row = choose_best_threshold(thresholds_df)

    thresholds_path = output_dir / "detect3_threshold_sweep.csv"
    best_path = output_dir / "detect3_best_threshold.csv"
    plot_path = output_dir / "detect3_threshold_sweep.png"

    thresholds_df.to_csv(thresholds_path, index=False, float_format="%.6f")
    pd.DataFrame([best_row]).to_csv(best_path, index=False, float_format="%.6f")
    plot_thresholds(thresholds_df, best_row, plot_path)

    print(f"Loaded scores: {input_csv}")
    print(f"Best threshold: {int(best_row['threshold'])}")
    print(f"Best precision: {best_row['precision']:.3f}")
    print(f"Best recall: {best_row['recall']:.3f}")
    print(f"Best F1: {best_row['f1']:.3f}")
    print(f"Saved threshold sweep: {thresholds_path}")
    print(f"Saved best threshold: {best_path}")
    print(f"Saved plot: {plot_path}")


if __name__ == "__main__":
    main()