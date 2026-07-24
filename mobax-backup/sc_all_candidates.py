# -*- coding: utf-8 -*-
"""Run the existing threshold-study SC algorithm on every candidate."""

import sc as base

from all_candidates import build_all_candidates


def main():
    args = base.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset_df = base.load_normalized_dataset(args.dataset)
    mt_candidates = build_all_candidates(args.dataset)
    cand_df = base.build_candidate_frame(
        mt_candidates,
        base.gold_by_sentence_and_type(dataset_df),
    )

    scoring_output = args.scoring_output or args.output_dir / "sc_scoring_breakdown.csv"
    thresholds_output = args.thresholds_output or args.output_dir / "sc_thr_metrics.csv"
    single_output = args.single_output or args.output_dir / "sc_preds_sw.csv"
    multi_output = args.multi_output or args.output_dir / "sc_preds_mw.csv"
    all_output = args.all_output or args.output_dir / "sc_preds_all.csv"
    metrics_output = args.metrics_output or args.output_dir / "sc_metrics_all.csv"
    cand_df.to_csv(scoring_output, index=False)

    threshold_df = base.evaluate_thresholds(cand_df)
    best = base.select_best_threshold(threshold_df)
    threshold_df.loc[
        threshold_df["threshold"].eq(best["threshold"]), "is_best"
    ] = 1
    applied_threshold = (
        base.load_selected_threshold(args.thresholds_input)
        if args.thresholds_input is not None
        else float(best["threshold"])
    )
    threshold_df["selected_for_prediction"] = (
        threshold_df["threshold"].eq(applied_threshold).astype(int)
    )
    threshold_df.to_csv(thresholds_output, index=False)

    final_df = base.apply_phrase_picker(
        base.apply_threshold(cand_df, applied_threshold)
    )
    final_df = final_df[base.output_columns(final_df)]
    single_df = final_df[
        final_df["cand_type"].astype(str).str.lower().eq("is_single_word")
    ].copy()
    multi_df = final_df[
        final_df["cand_type"].astype(str).str.lower().eq("is_multiword")
    ].copy()

    single_df.to_csv(single_output, index=False)
    multi_df.to_csv(multi_output, index=False)
    final_df.to_csv(all_output, index=False)
    base.write_metrics(final_df, metrics_output)

    print(f"Wrote scoring breakdown to {scoring_output}", flush=True)
    print(f"Wrote threshold metrics to {thresholds_output}", flush=True)
    print(f"Wrote single-word predictions to {single_output}", flush=True)
    print(f"Wrote multiword predictions to {multi_output}", flush=True)
    print(f"Wrote all predictions to {all_output}", flush=True)
    print(f"Wrote metrics to {metrics_output}", flush=True)


if __name__ == "__main__":
    main()
