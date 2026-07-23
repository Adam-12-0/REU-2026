# -*- coding: utf-8 -*-

import argparse
import unicodedata
from pathlib import Path

import pandas as pd
import regex as re
from rapidfuzz import fuzz
from wordfreq import zipf_frequency


DEFAULT_OUTPUT_DIR = Path("surfaceclues_outputs")
THRESHOLDS = [round(x / 100, 2) for x in range(10, 71, 10)]
WORD_RE = re.compile(r"[\p{L}\p{M}]+(?:['\u2019-][\p{L}\p{M}]+)?|[\p{N}]+")
REQUIRED_DATASET_COLUMNS = [
    "sentence",
    "gold_terms",
    "term_type",
    "source_dataset",
    "is_single_word",
    "is_multiword",
]
APOSTROPHES = str.maketrans({
    "\u2018": "'",
    "\u2019": "'",
    "\u201a": "'",
    "\u201b": "'",
    "\u2032": "'",
    "\u2035": "'",
    "`": "'",
    "\u00b4": "'",
})
HYPHENS = str.maketrans({
    "\u2010": "-",
    "\u2011": "-",
    "\u2012": "-",
    "\u2013": "-",
    "\u2014": "-",
    "\u2015": "-",
    "\u2212": "-",
})


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Surface Clues over MT-proposed candidates."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--mt-input",
        type=Path,
        default=None,
        help="MT all-predictions file. Defaults to output_dir/../mt_outputs/mt_preds_all.csv.",
    )
    parser.add_argument("--scoring-output", type=Path, default=None)
    parser.add_argument("--thresholds-output", type=Path, default=None)
    parser.add_argument(
        "--thresholds-input",
        type=Path,
        default=None,
        help="Optional calibration threshold table to apply instead of tuning on this dataset.",
    )
    parser.add_argument("--single-output", type=Path, default=None)
    parser.add_argument("--multi-output", type=Path, default=None)
    parser.add_argument("--all-output", type=Path, default=None)
    parser.add_argument("--metrics-output", type=Path, default=None)
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


def deterministic_normalize(text, keep_internal_punctuation=True):
    text = unicodedata.normalize("NFKC", str(text))
    text = text.translate(APOSTROPHES).translate(HYPHENS)
    text = text.casefold().strip()
    text = re.sub(r"\s+", " ", text)
    if keep_internal_punctuation:
        text = re.sub(r"[^\p{L}\p{N}'\-\s]+", " ", text)
    else:
        text = re.sub(r"[^\p{L}\p{N}\s]+", " ", text)
    text = re.sub(r"^[^\p{L}\p{N}]+|[^\p{L}\p{N}]+$", "", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_term(text):
    return deterministic_normalize(text, keep_internal_punctuation=True)


def normalize_word(text):
    return deterministic_normalize(text, keep_internal_punctuation=True).replace(" ", "")


def tokenize_words(text):
    return [match.group(0) for match in WORD_RE.finditer(str(text))]


def normalize_candidate(row):
    cand_type = str(row.get("cand_type", "")).lower()
    if cand_type == "is_single_word":
        return normalize_word(row.get("cand_norm", row.get("candidate", "")))
    return normalize_term(row.get("cand_norm", row.get("candidate", "")))


def term_zipf(term):
    words = re.findall(r"[\p{L}]+", str(term).casefold())
    if not words:
        return 0.0
    return min(zipf_frequency(word, "en") for word in words)


def surface_features(candidate):
    text = str(candidate).strip()
    lower = text.casefold()
    zipf = term_zipf(text)
    return {
        "zipf": zipf,
        "rare_or_unknown": int(zipf == 0 or zipf < 3.0),
        "is_acronym": int(bool(re.fullmatch(r"[A-Z]{2,}", text))),
        "has_digit": int(bool(re.search(r"\p{N}", text))),
        "has_symbol": int(bool(re.search(r"[^\p{L}\p{N}\s'\u2019\-]", text))),
        "has_repeated_char": int(bool(re.search(r"(.)\1{2,}", lower))),
    }


def surface_score(features):
    raw = (
        1.5 * features["rare_or_unknown"]
        + 1.0 * features["is_acronym"]
        + 0.5 * features["has_digit"]
        + 0.5 * features["has_symbol"]
        + 0.5 * features["has_repeated_char"]
    )
    return min(raw / 3.5, 1.0)


def is_gold_candidate(candidate_norm, gold_norms):
    for gold_norm in gold_norms:
        if candidate_norm == gold_norm or fuzz.ratio(candidate_norm, gold_norm) >= 92:
            return 1
    return 0


def load_normalized_dataset(path):
    df = pd.read_csv(path)
    require_columns(df, REQUIRED_DATASET_COLUMNS, path)
    df = df.dropna(subset=["sentence", "gold_terms"]).copy()
    df["sentence"] = df["sentence"].astype(str)
    df["gold_norm"] = df["gold_terms"].apply(normalize_term)
    df["is_single_word"] = df["is_single_word"].apply(to_binary)
    df["is_multiword"] = df["is_multiword"].apply(to_binary)
    return df


def sentence_id_map(dataset_df):
    return {
        sentence: index
        for index, sentence in enumerate(dataset_df["sentence"].drop_duplicates().tolist())
    }


def gold_by_sentence_and_type(dataset_df):
    ids = sentence_id_map(dataset_df)
    output = {}
    for sentence, group in dataset_df.groupby("sentence", sort=False):
        sentence_id = str(ids[sentence])
        output[(sentence_id, "is_single_word")] = {
            normalize_word(value)
            for value in group.loc[group["is_single_word"].eq(1), "gold_norm"]
            if value
        }
        output[(sentence_id, "is_multiword")] = {
            normalize_term(value)
            for value in group.loc[group["is_multiword"].eq(1), "gold_norm"]
            if value
        }
    return output


def load_mt_candidates(mt_input):
    mt_df = pd.read_csv(mt_input)
    if "candidate_type" in mt_df.columns and "cand_type" not in mt_df.columns:
        mt_df = mt_df.rename(columns={"candidate_type": "cand_type"})
    require_columns(
        mt_df,
        ["sentence_id", "sentence", "candidate", "cand_norm", "cand_type", "mt_pred"],
        mt_input,
    )
    mt_df = mt_df.copy()
    mt_df["cand_type"] = mt_df["cand_type"].astype(str)
    mt_df["cand_norm"] = mt_df.apply(normalize_candidate, axis=1)
    return mt_df


def candidate_component_positions(row):
    if str(row.get("cand_type", "")).lower() != "is_multiword":
        return ""
    try:
        start = int(float(row.get("token_pos", "")))
    except (TypeError, ValueError):
        return ""
    token_count = len(tokenize_words(row.get("candidate", "")))
    if token_count < 2:
        token_count = len(tokenize_words(row.get("cand_norm", "")))
    if token_count < 2:
        return ""
    return "|".join(str(pos) for pos in range(start, start + token_count))


def build_candidate_frame(mt_df, gold_lookup):
    columns = [
        "sentence_id",
        "sentence",
        "candidate",
        "cand_norm",
        "cand_type",
        "is_gold",
        "token_pos",
        "component_token_positions",
        "mt_pred",
        "sc_score",
        "sc_pred",
    ]
    if mt_df.empty:
        return pd.DataFrame(columns=columns)

    cand_df = mt_df.copy()
    cand_df["is_gold"] = cand_df.apply(
        lambda row: is_gold_candidate(
            row["cand_norm"],
            gold_lookup.get((str(row["sentence_id"]), str(row["cand_type"]).lower()), set()),
        ),
        axis=1,
    )
    cand_df["component_token_positions"] = cand_df.apply(candidate_component_positions, axis=1)

    score_mask = cand_df["mt_pred"].apply(to_binary).eq(1)
    feature_rows = cand_df["candidate"].apply(surface_features).apply(pd.Series)
    cand_df = pd.concat([cand_df, feature_rows], axis=1)
    cand_df["sc_score"] = 0.0
    cand_df.loc[score_mask, "sc_score"] = cand_df.loc[score_mask].apply(surface_score, axis=1)
    cand_df["sc_pred"] = 0
    return cand_df


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


def apply_threshold(cand_df, threshold):
    pred_df = cand_df.copy()
    score_mask = pred_df["mt_pred"].apply(to_binary).eq(1)
    pred_df["sc_pred"] = 0
    pred_df.loc[score_mask, "sc_pred"] = (
        pred_df.loc[score_mask, "sc_score"].astype(float).ge(float(threshold))
    ).astype(int)
    return pred_df


def parse_component_positions(value):
    positions = []
    for item in str(value).split("|"):
        if item == "":
            continue
        try:
            positions.append(int(item))
        except ValueError:
            continue
    return positions


def apply_phrase_picker(pred_df):
    final_df = pred_df.copy()
    final_df["_numeric_token_pos"] = pd.to_numeric(final_df["token_pos"], errors="coerce")
    phrase_df = final_df[
        final_df["cand_type"].astype(str).str.lower().eq("is_multiword")
        & final_df["sc_pred"].astype(int).eq(1)
    ].copy()
    if phrase_df.empty:
        return final_df.drop(columns=["_numeric_token_pos"])

    phrase_df["_length"] = phrase_df["component_token_positions"].apply(
        lambda value: len(parse_component_positions(value))
    )
    claimed_positions = set()

    for phrase_index, phrase_row in phrase_df.sort_values(
        ["sc_score", "_length"],
        ascending=[False, False],
        kind="mergesort",
    ).iterrows():
        sentence_id = str(phrase_row["sentence_id"])
        component_positions = parse_component_positions(phrase_row["component_token_positions"])
        if not component_positions or any(
            (sentence_id, position) in claimed_positions for position in component_positions
        ):
            final_df.loc[phrase_index, "sc_pred"] = 0
            continue

        word_mask = (
            final_df["sentence_id"].astype(str).eq(sentence_id)
            & final_df["cand_type"].astype(str).str.lower().eq("is_single_word")
            & final_df["_numeric_token_pos"].isin(component_positions)
        )
        max_word_score = final_df.loc[word_mask, "sc_score"].astype(float).max()
        if pd.notna(max_word_score) and float(phrase_row["sc_score"]) > float(max_word_score):
            final_df.loc[phrase_index, "sc_pred"] = 1
            final_df.loc[word_mask, "sc_pred"] = 0
            for position in component_positions:
                claimed_positions.add((sentence_id, position))
        else:
            final_df.loc[phrase_index, "sc_pred"] = 0

    return final_df.drop(columns=["_numeric_token_pos"])


def metric_row(cand_type, df):
    y_true = df["is_gold"].astype(int)
    y_pred = df["sc_pred"].astype(int)
    tp, fp, fn, _tn = calculate_counts(y_true, y_pred)
    precision, recall, f1 = calculate_metrics(tp, fp, fn)
    return {
        "cand_type": cand_type,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def write_metrics(df, output_path):
    rows = [metric_row(cand_type, group) for cand_type, group in df.groupby("cand_type", sort=True)]
    rows.append(metric_row("all", df))
    pd.DataFrame(rows).to_csv(output_path, index=False)


def evaluate_thresholds(cand_df):
    rows = []
    y_true = cand_df["is_gold"].astype(int)
    for threshold in THRESHOLDS:
        pred_df = apply_phrase_picker(apply_threshold(cand_df, threshold))
        y_pred = pred_df["sc_pred"].astype(int)
        tp, fp, fn, tn = calculate_counts(y_true, y_pred)
        precision, recall, f1 = calculate_metrics(tp, fp, fn)
        rows.append(
            {
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
                "candidate_count": int(len(cand_df)),
                "is_best": 0,
            }
        )
    return pd.DataFrame(rows)


def select_best_threshold(threshold_df):
    return threshold_df.sort_values(
        ["f1", "precision", "recall", "threshold"],
        ascending=[False, False, False, True],
    ).iloc[0]


def load_selected_threshold(path):
    threshold_df = pd.read_csv(path)
    require_columns(threshold_df, ["threshold"], path)
    for column in ("selected_for_prediction", "is_best"):
        if column in threshold_df.columns:
            selected = threshold_df[threshold_df[column].apply(to_binary).eq(1)]
            if not selected.empty:
                return float(selected.iloc[0]["threshold"])
    if threshold_df.empty:
        raise ValueError(f"No threshold found in {path}")
    return float(select_best_threshold(threshold_df)["threshold"])


def output_columns(df):
    preferred = [
        "sentence_id",
        "sentence",
        "candidate",
        "cand_norm",
        "cand_type",
        "is_gold",
        "token_pos",
        "component_token_positions",
        "mt_pred",
        "sc_score",
        "sc_pred",
    ]
    columns = [column for column in preferred if column in df.columns]
    columns.extend(column for column in df.columns if column not in columns)
    return columns


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset_df = load_normalized_dataset(args.dataset)
    mt_input = args.mt_input
    if mt_input is None:
        mt_input = args.output_dir.parent / "mt_outputs" / "mt_preds_all.csv"

    mt_df = load_mt_candidates(mt_input)
    cand_df = build_candidate_frame(mt_df, gold_by_sentence_and_type(dataset_df))

    scoring_output = args.scoring_output or args.output_dir / "sc_scoring_breakdown.csv"
    thresholds_output = args.thresholds_output or args.output_dir / "sc_thr_metrics.csv"
    single_output = args.single_output or args.output_dir / "sc_preds_sw.csv"
    multi_output = args.multi_output or args.output_dir / "sc_preds_mw.csv"
    all_output = args.all_output or args.output_dir / "sc_preds_all.csv"
    metrics_output = args.metrics_output or args.output_dir / "sc_metrics_all.csv"
    cand_df.to_csv(scoring_output, index=False)

    threshold_df = evaluate_thresholds(cand_df)
    best = select_best_threshold(threshold_df)
    threshold_df.loc[threshold_df["threshold"].eq(best["threshold"]), "is_best"] = 1
    applied_threshold = (
        load_selected_threshold(args.thresholds_input)
        if args.thresholds_input is not None
        else float(best["threshold"])
    )
    threshold_df["selected_for_prediction"] = threshold_df["threshold"].eq(applied_threshold).astype(int)
    threshold_df.to_csv(thresholds_output, index=False)

    final_df = apply_phrase_picker(apply_threshold(cand_df, applied_threshold))
    columns = output_columns(final_df)
    final_df = final_df[columns]

    single_df = final_df[final_df["cand_type"].astype(str).str.lower().eq("is_single_word")].copy()
    multi_df = final_df[final_df["cand_type"].astype(str).str.lower().eq("is_multiword")].copy()

    single_df.to_csv(single_output, index=False)
    multi_df.to_csv(multi_output, index=False)
    final_df.to_csv(all_output, index=False)
    write_metrics(final_df, metrics_output)

    print(f"Wrote scoring breakdown to {scoring_output}", flush=True)
    print(f"Wrote threshold metrics to {thresholds_output}", flush=True)
    print(f"Wrote single-word predictions to {single_output}", flush=True)
    print(f"Wrote multiword predictions to {multi_output}", flush=True)
    print(f"Wrote all predictions to {all_output}", flush=True)
    print(f"Wrote metrics to {metrics_output}", flush=True)


if __name__ == "__main__":
    main()
