# MT-First Specialized Language Detection Pipeline

This directory contains the active end-to-end pipeline for detecting specialized words and phrases. It is designed for a Slurm cluster and uses Machine Translation as a high-recall proposal stage before applying three more expensive downstream detectors.

## Architecture

The pipeline follows five stages:

1. Machine Translation scores every word in each input sentence.
2. The selected MT threshold identifies specialized-word proposals.
3. Consecutive MT-positive words are combined into multiword phrase candidates.
4. Surface Clues, Likelihood Gap, and Model Agreement rescore the proposed words and phrases in parallel.
5. Logistic fusion combines the three downstream binary predictions and resolves word-versus-phrase conflicts.

```text
                         +--> Surface Clues ----+
                         |                       |
Input --> MT words --> word and phrase candidates --> Logistic fusion --> predictions
                         |                       |
                         +--> Likelihood Gap ----+
                         |                       |
                         +--> Model Agreement ---+
```

Machine Translation is an upstream gate, not a fourth logistic feature. The current fusion model combines `sc_pred`, `lg_pred`, and `ma_pred` after MT has defined the candidate space.

## Candidate flow

### Word candidates

The MT stage tokenizes each sentence and creates one row per word occurrence. Each row receives:

- a four-language translation-instability score,
- a selected MT prediction,
- a sentence identifier and token position,
- a gold label for evaluation.

The word output retains MT-positive and MT-negative rows. Downstream methods score only MT-positive rows and leave MT-negative rows at zero. Retaining all word rows allows MT misses to remain visible during word-level evaluation.

### Phrase candidates

`mt_consecutive.py` scans each sentence in token order. A run continues while adjacent words are MT-positive and are not separated by a comma. Every unique contiguous subspan of at least two words within the run becomes a phrase candidate.

For example, if `touch`, `some`, and `grass` are consecutive MT-positive words, the generator can produce:

- `touch some`
- `some grass`
- `touch some grass`

Generated phrase rows receive `mt_pred = 1` and are passed to all three downstream methods together with the word rows.

## Pipeline stages

| Stage | Scripts | Function | Main outputs |
| --- | --- | --- | --- |
| MT | `mt_judge.py`, `mt_judge_en.py`, `mt_judge_predictions.py`, `mt_consecutive.py` | Score words with four round-trip translations, select MT proposals, and generate phrases | `mt_preds_sw.csv`, `mt_preds_mw.csv`, `mt_preds_all.csv` |
| SC | `sc.py` | Score MT candidates using surface-form features | `sc_preds_all.csv`, `sc_metrics_all.csv` |
| LG | `lg.py`, `lg_pred.py` | Score MT candidates using XLM-RoBERTa likelihood gap | `lg_preds_all.csv`, `lg_metrics_all.csv` |
| MA | `ma_agree.py`, `ma_score.py`, `ma_pred.py` | Generate three contextual definitions and score their agreement | `ma_preds_all.csv`, `ma_metrics_all.csv` |
| LF | `compile.py`, `lf.py` | Merge downstream predictions, train logistic regression, and select final spans | `preds_compiled.csv`, `lf_preds_all.csv`, `lf_metrics_all.csv` |

## Models and signals

### Machine Translation

Machine Translation uses Meta Llama 3.1 8B Instruct to translate each sentence through Spanish, Arabic, Chinese, and Japanese round trips. Each language votes that a word is specialized when the best fuzzy match in the back-translation falls below the configured threshold. A word that remains unchanged across translation is also treated as specialized unless spaCy identifies it as a named entity.

### Surface Clues

Surface Clues uses word frequency and orthographic features, including acronyms, digits, symbols, and repeated characters. Only MT-positive candidates receive nonzero Surface Clues scores.

### Likelihood Gap

Likelihood Gap uses `xlm-roberta-base`. It masks the candidate span and compares the candidate-token likelihood with the model's most likely replacement. Only MT-positive candidates are scored.

### Model Agreement

Model Agreement asks three instruction-tuned models for context-specific definitions:

- Meta Llama 3.1 8B Instruct
- Qwen 2.5 7B Instruct
- Mistral 7B Instruct v0.3

The pairwise definitions are scored with `cross-encoder/stsb-roberta-large`. Lower semantic similarity is intended to indicate lower agreement and therefore a more specialized candidate.

### Logistic fusion

The current logistic model uses three binary features:

- `sc_pred`
- `lg_pred`
- `ma_pred`

It does not currently use the continuous detector scores or `mt_score`. Logistic probabilities are thresholded, then a phrase picker resolves overlapping words and phrases.

## Input schema

The input CSV must contain:

| Column | Type | Description |
| --- | --- | --- |
| `sentence` | text | Source sentence |
| `gold_terms` | text | Annotated specialized term or phrase |
| `term_type` | text | Annotation category |
| `source_dataset` | text | Dataset identifier |
| `is_single_word` | binary | One for a word annotation |
| `is_multiword` | binary | One for a phrase annotation |

Rows with the same sentence may contain separate gold terms. Sentence identifiers are assigned from the first-occurrence order of unique sentence text.

## Requirements

The provided job files assume:

- a Slurm cluster,
- Bash,
- the `anaconda/anaconda-2024.10` module,
- a conda environment named `dlcv`,
- CUDA-capable GPU nodes,
- access to the configured Hugging Face models,
- a compatible spaCy English named-entity model,
- Python packages used by the scripts, including PyTorch, Transformers, pandas, scikit-learn, sentence-transformers, spaCy, rapidfuzz, regex, NumPy, and wordfreq.

The repository does not currently provide a pinned environment file. Reproducible runs should record package versions, model revisions, random seeds, and the exact dataset revision.

## Configuration

Edit the variables near the top of `run_driver.slurm`:

```bash
DATASET="genz_b_normalized.csv"
NAME="genz_b_rp"
SOURCE_IS_ENGLISH="1"
OUTDIR="pipeline_outputs/${NAME}"
```

| Variable | Description |
| --- | --- |
| `DATASET` | Input CSV path, relative to the submission directory unless absolute |
| `NAME` | Run identifier used in job names, logs, and output paths |
| `SOURCE_IS_ENGLISH` | Use `1` for English input and `0` for non-English input |
| `OUTDIR` | Root directory for all outputs from the run |

Use a unique `NAME` for each experiment to avoid mixing outputs from different datasets or configurations.

## Running the pipeline

Run from the directory containing the scripts:

```bash
sbatch run_driver.slurm
```

The driver is a short submission job. Do not launch it with `srun`.

The dependency graph is:

```text
MT
|-- after success --> SC --+
|-- after success --> LG --+--> after all succeed --> LF
|-- after success --> MA --+
```

The driver uses Slurm `afterok` dependencies:

- SC, LG, and MA start only after MT succeeds.
- LF starts only after SC, LG, and MA all succeed.
- A failed upstream stage prevents its dependent stage from starting.

## Output layout

For `OUTDIR="pipeline_outputs/${NAME}"`, outputs are organized as:

```text
pipeline_outputs/${NAME}/
|-- logs/
|-- mt_outputs/
|-- sc_outputs/
|-- lg_outputs/
|-- ma_outputs/
`-- lf_outputs/
```

### MT outputs

| File | Description |
| --- | --- |
| `thr_metrics.csv` | MT word-threshold sweep and selected threshold |
| `mt_preds_sw.csv` | All word candidates and MT predictions |
| `mt_preds_mw.csv` | Consecutive phrase candidates |
| `mt_preds_all.csv` | Combined word and phrase candidate table |
| `mt_metrics_all.csv` | Word, phrase, and combined candidate-level metrics |

The temporary `_mt_word_scores_tmp.csv` file is removed after successful processing.

### SC outputs

| File | Description |
| --- | --- |
| `sc_scoring_breakdown.csv` | Surface features and scores for all MT candidates |
| `sc_thr_metrics.csv` | Surface Clues threshold sweep |
| `sc_preds_sw.csv` | Word predictions |
| `sc_preds_mw.csv` | Phrase predictions |
| `sc_preds_all.csv` | Combined predictions |
| `sc_metrics_all.csv` | Candidate-level metrics |

### LG outputs

| File | Description |
| --- | --- |
| `lg_scoring_breakdown.csv` | Likelihood features and scores |
| `lg_thr_metrics.csv` | Likelihood Gap threshold sweep |
| `lg_preds_sw.csv` | Word predictions |
| `lg_preds_mw.csv` | Phrase predictions |
| `lg_preds_all.csv` | Combined predictions |
| `lg_metrics_all.csv` | Candidate-level metrics |

### MA outputs

| File | Description |
| --- | --- |
| `ma_thr_metrics.csv` | Mean-similarity threshold sweep |
| `ma_preds_sw.csv` | Word predictions |
| `ma_preds_mw.csv` | Phrase predictions |
| `ma_preds_all.csv` | Combined predictions |
| `ma_metrics_all.csv` | Candidate-level metrics |

The temporary definition and score files are removed after successful processing.

### LF outputs

| File | Description |
| --- | --- |
| `preds_compiled.csv` | Merged SC, LG, and MA candidate predictions |
| `lf_preds_sw.csv` | Final word predictions |
| `lf_preds_mw.csv` | Final phrase predictions |
| `lf_preds_all.csv` | Final combined predictions and fusion scores |
| `lf_thr_metrics.csv` | Logistic threshold sweep |
| `lf_weights.csv` | Learned coefficients and intercept |
| `lf_metrics_all.csv` | Final candidate-level metrics |

## Rerunning failed stages

The repository includes:

- `rerun_sc.slurm`
- `rerun_ma.slurm`
- `rerun_lf.slurm`

Use the same `DATASET`, `NAME`, and `OUTDIR` values as the original run. If MT fails, rerun the full driver because every downstream stage depends on MT. If SC or MA fails, allow the other independent stages to finish, rerun the failed stage, then submit LF after all required outputs exist. LG does not currently have a dedicated rerun wrapper, so rerun `run_lg.slurm` with the required exported variables or rerun the driver with a new run name.

## Evaluation semantics

Current metric files are candidate-level evaluations over the rows present in each stage output. Word rows include MT-positive and MT-negative candidates. Phrase rows include only phrases generated from consecutive MT-positive words.

This distinction matters. Candidate-level phrase recall is conditional on phrase generation and is not yet a valid end-to-end span recall measure. A full evaluator should outer join predictions against every annotated gold span so ungenerated phrases count as false negatives.

Threshold selection and logistic training currently use the same dataset rows that are reported in the metric files. These numbers are development metrics, not held-out test metrics.

## Known correctness issues

The following issues should be fixed before publishing or comparing final results:

1. `ma_pred.py` applies the selected Model Agreement threshold to `ma_score = 1 - mean_similarity` with the wrong inequality. It should apply `mean_similarity <= threshold`, or the equivalent transformed rule.
2. Ungenerated gold phrases are missing from the phrase recall denominator.
3. Thresholds and logistic regression are selected and evaluated in sample.
4. Logistic fusion uses three binary predictions, which creates at most eight input patterns and can cause majority, weighted, and logistic rules to collapse to identical predictions.
5. Candidate merge keys are not validated as one-to-one.
6. MT vote scores are min-max normalized separately for each dataset, reducing threshold comparability across datasets.

## Recommended evaluation protocol

1. Split data by sentence, source dataset, or term family before any tuning.
2. Fit detector calibration and logistic coefficients on training data.
3. Select all thresholds on validation data.
4. Evaluate once on a locked test set with a complete gold-span denominator.
5. Save keyed predictions for every fusion method.
6. Report pairwise prediction disagreements alongside TP, FP, FN, TN, precision, recall, and F1.

## Troubleshooting

### MT fails

Check GPU allocation, Hugging Face model access, the spaCy model, dataset schema, and the MT log file. Downstream jobs will remain blocked because of `afterok`.

### A downstream stage fails

Inspect the stage-specific error log under `logs/`. Confirm that `mt_preds_all.csv` exists and that the rerun uses the same dataset and output directory.

### LF fails during compilation

Verify that `sc_preds_all.csv`, `lg_preds_all.csv`, and `ma_preds_all.csv` all exist and describe the same run. Check that candidate identifiers, candidate types, normalized candidate text, and token positions are consistent.

### Fusion methods report identical metrics

Do not compare only aggregate metrics. Join outputs by `sentence_id`, `cand_type`, `cand_norm`, and `token_pos`, then count pairwise prediction disagreements. If the prediction vectors are identical, inspect the number and frequency of unique downstream prediction patterns. With three binary features, no more than eight patterns are possible.
