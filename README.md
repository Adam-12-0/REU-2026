# REU 2026

This repository supports the REU 2026 project on detecting and defining specialized language in text.

The current work focuses on identifying candidate terms that may be slang, domain-specific, unstable across translation, or otherwise specialized. The project uses multiple scoring methods so candidates can be compared across complementary signals instead of relying on a single detector.

## Project Focus

The initial methods developed so far are:

- Surface clues: scores candidates using features such as frequency, acronyms, digits, symbols, emojis, repeated characters, and corpus behavior.
- Language surprise: masks candidate words or phrases and uses XLM-RoBERTa fill-mask likelihood gaps to estimate whether a token is unexpected in context.
- Machine translation instability: translates text out of English and back into English, then uses fuzzy matching and named entity checks to detect terms that changed, disappeared, or remained untranslated.

The working dataset referenced in the weekly updates is `acader/genz-alpha-slangs`.

## Current Status

The repository is being set up as a shared workspace for the mentoring project. Code will be added by the student, and this README can be expanded later with:

- installation instructions
- dataset setup
- model and environment requirements
- commands for running each method
- evaluation results
- examples of expected inputs and outputs

## Project Notes

The week 3 through week 5 updates describe progress on the first three methods, including threshold exploration, candidate type experiments, multilingual translation tests, and early comparisons between method performance.

## Contributors

- Valerie Lopez
- Adam Bawatneh
- Dr. Santu Karmaker
- Dr. Song Wang
- Dr. Mubarak Shah
