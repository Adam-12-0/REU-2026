# -*- coding: utf-8 -*-
"""Run the existing threshold-study LG algorithm on every candidate."""

import lg as base

from all_candidates import build_all_candidates


def main():
    args = base.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    features_output = (
        args.features_output
        or args.output_dir / "lg_scoring_breakdown.csv"
    )
    thresholds_output = (
        args.thresholds_output
        or args.output_dir / "lg_thr_metrics.csv"
    )

    print("Starting all-candidate likelihood-gap run", flush=True)
    print(f"Dataset: {args.dataset}", flush=True)
    print(f"Output directory: {args.output_dir}", flush=True)
    print(f"Model name: {args.model_name}", flush=True)

    base.torch.cuda.is_available()
    tokenizer = base.AutoTokenizer.from_pretrained(args.model_name)
    device = base.torch.device(
        "cuda" if base.torch.cuda.is_available() else "cpu"
    )
    print("Device:", device, flush=True)
    if base.torch.cuda.is_available():
        print("GPU:", base.torch.cuda.get_device_name(0), flush=True)
    model = base.AutoModelForMaskedLM.from_pretrained(args.model_name).to(device)
    model.eval()

    dataset_df = base.load_normalized_dataset(args.dataset)
    candidates_df = build_all_candidates(args.dataset)
    scored_df = base.score_mt_candidates(
        candidates_df,
        tokenizer,
        model,
        base.gold_by_sentence_and_type(dataset_df),
    )
    scored_df.to_csv(features_output, index=False)
    print(f"Wrote {len(scored_df)} candidate scores to {features_output}", flush=True)

    thresholds_df = base.evaluate_thresholds(scored_df, args.num_thresholds)
    thresholds_df.to_csv(thresholds_output, index=False)
    print(f"Wrote threshold metrics to {thresholds_output}", flush=True)


if __name__ == "__main__":
    main()
