""" Receive three sets of definitions from three LLMs. For each candidate, review its three definitions and
score mean semantic agreement and disagreement.
"""
import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

DEFAULT_SIMILARITY_MODEL_ID = "sentence-transformers/all-mpnet-base-v2"


# Calculate cosine similarity of two embedding arrays
def cosine_rows(left, right):
    numerator = np.sum(left * right, axis=1)
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=float),
        where=denominator != 0
    )


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", type=Path, default=Path("_ma_definitions_tmp.csv"))
    parser.add_argument("--output-scores", type=Path, default=Path("_ma_word_scores_tmp.csv"))
    
    parser.add_argument("--embedding-model", default=DEFAULT_SIMILARITY_MODEL_ID)
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_scores.parent.mkdir(parents=True, exist_ok=True)

    # Load candidate definitions
    df = pd.read_csv(args.input)
   
    # Store definitions into a separate DataFrame
    definition_columns = [
        column for column in df.columns if column.endswith("_definition")
    ]
    definitions = df[definition_columns].fillna("").astype(str)
    
    # Load model for calculating semantic text similarity
    model = SentenceTransformer(args.embedding_model)
    
    # Create embeddings for each model's definition
    embeddings = {}
    for col in definition_columns:
        definition_texts = definitions[col].tolist() # Converts column of definitions to Python list

        column_embeddings = model.encode(
            definition_texts,
            batch_size=args.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        embeddings[col] = column_embeddings # Save vectors under definition column's name

    # Compare every possible unique pairing by semantic similarity
    similarity_columns = [] # Similarity pair columns: ex. llama-qwen
    for left, right in combinations(definition_columns, 2):
        # Save model pair names
        suffix = "_definition"
        model1_name = left[:-len(suffix)]
        model2_name = right[:-len(suffix)]
        pair_name = f"{model1_name}_{model2_name}_similarity"

        # Calculate cosine similarities
        df[pair_name] = cosine_rows(embeddings[left], embeddings[right])
        similarity_columns.append(pair_name)

    # Print average similarity across columns (all possible pairs)
    df["mean_similarity"] = df[similarity_columns].mean(axis=1) # Agreement
    df["disagreement_score"] = 1.0 - df["mean_similarity"] # Disagreement
    df.to_csv(args.output_scores, index=False)
    print(f"Wrote word scores to {args.output_scores}", flush=True)


if __name__ == "__main__":
    main()
