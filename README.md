# REU 2026: Detecting Specialized Language in Text

This repository contains an experimental NLP pipeline for finding specialized language in context, including slang, jargon, acronyms, complex terms, idioms, and multiword expressions.

The current system uses a Machine Translation first, or MT-first, architecture. Machine Translation proposes specialized words and consecutive phrase candidates. Three downstream detectors then rescore those candidates, and a logistic model combines the downstream decisions.

## Current pipeline

The active implementation is in [`mt-first-pipeline`](mt-first-pipeline/README.md).

```text
Input sentences
      |
      v
Machine Translation word detection
      |
      +--> all word candidates
      |
      +--> consecutive phrases from MT-positive words
                    |
                    v
        Surface Clues, Likelihood Gap,
        and Model Agreement in parallel
                    |
                    v
             Logistic fusion
                    |
                    v
       Final word and phrase predictions
```

Machine Translation serves two purposes:

1. It produces the first specialized-word predictions.
2. It limits expensive downstream processing to promising words and phrases.

The downstream fusion stage currently combines three binary reranker predictions: Surface Clues, Likelihood Gap, and Model Agreement. Machine Translation is the proposal gate and is not included again as a logistic-regression feature.

## Detection methods

| Method | Signal | Current role |
| --- | --- | --- |
| Machine Translation | Instability across four round-trip translations, fuzzy matching, and named-entity checks | Candidate proposal and phrase generation |
| Surface Clues | Rarity, capitalization, digits, symbols, and repeated characters | Downstream candidate scoring |
| Likelihood Gap | XLM-RoBERTa masked-language-model surprise | Downstream candidate scoring |
| Model Agreement | Disagreement among contextual definitions from three language models | Downstream candidate scoring |

## Repository layout

| Path | Purpose |
| --- | --- |
| `mt-first-pipeline/` | Active end-to-end Slurm pipeline |
| `surface-clues/` | Earlier standalone Surface Clues experiments |
| `likelihood-gap/` | Earlier standalone Likelihood Gap experiments and saved results |
| `mach-translation/` | Earlier language-specific, judge, and layered MT experiments |
| `model-agree/` | Earlier standalone Model Agreement experiments and saved results |

The standalone folders document the development of each detector. They are useful for historical experiments, but they are not the current end-to-end pipeline.

## Input data

The MT-first pipeline expects a normalized CSV with these columns:

| Column | Meaning |
| --- | --- |
| `sentence` | Source sentence |
| `gold_terms` | Annotated specialized term or phrase |
| `term_type` | Annotation category |
| `source_dataset` | Dataset identifier |
| `is_single_word` | Binary flag for a one-word annotation |
| `is_multiword` | Binary flag for a multiword annotation |

Several normalized research datasets are included in `mt-first-pipeline/`. See the pipeline README for configuration and execution instructions.

## Current evaluation status

The pipeline is research code and its reported metrics should be treated as provisional. The current audit identified issues that must be corrected before results are presented as held-out test performance:

1. The final Model Agreement threshold is applied in the wrong score direction.
2. Gold phrases that are not generated can be omitted from the recall denominator.
3. Detector thresholds and logistic fusion are selected and evaluated on the same rows.
4. Binary downstream features create very few fusion input patterns, which can make different fusion rules produce identical predictions.

These issues affect evaluation and fusion behavior, not the intended MT-first architecture itself.

## Running the project

The current pipeline targets a Slurm cluster with GPU nodes and a configured Python environment. Start with the detailed instructions in [`mt-first-pipeline/README.md`](mt-first-pipeline/README.md).

## Project history

The weekly presentations document the progression from standalone detectors to consecutive phrase generation, fusion experiments, and the MT-first pipeline. Week 7 introduces consecutive candidates and fusion. Week 8 presents the explicit MT-first execution flow.

## Contributors

- Valerie Lopez
- Adam Bawatneh
- Dr. Santu Karmaker
- Dr. Song Wang
- Dr. Mubarak Shah
