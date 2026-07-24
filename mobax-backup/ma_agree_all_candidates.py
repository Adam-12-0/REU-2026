# -*- coding: utf-8 -*-
"""Generate MA definitions for every candidate using the existing MA models."""

import os
import ma_agree as base

from all_candidates import build_all_candidates


def main():
    args = base.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset_df = base.load_normalized_dataset(args.dataset, args.limit_rows)
    candidates_df = build_all_candidates(
        args.dataset,
        limit_sentences=args.limit_rows,
    )
    gold_lookup = base.gold_by_sentence_and_type(dataset_df)
    candidates_df["cand_type"] = candidates_df["cand_type"].astype(str)
    candidates_df["cand_norm"] = candidates_df.apply(
        base.normalize_candidate,
        axis=1,
    )
    candidates_df["is_gold"] = candidates_df.apply(
        lambda row: base.is_gold_candidate(
            row["cand_norm"],
            gold_lookup.get(
                (str(row["sentence_id"]), str(row["cand_type"]).lower()),
                set(),
            ),
        ),
        axis=1,
    )

    definitions_output = (
        args.definitions_output
        or args.output_dir / "_ma_definitions_tmp.csv"
    )
    definitions_df = base.define_candidates(
        candidates_df,
        args.model_ids,
        base.parse_gpu_ids(args.gpu_ids),
        args.batch_size,
        definitions_output,
    )
    definitions_df.to_csv(definitions_output, index=False)
    print(f"Saved temporary definitions: {definitions_output}", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
