import argparse
from pathlib import Path

import numpy as np

from imdb_chatbot import EMBEDDING_MODEL, embed, load_chunks


# Each question has a known answer in the export. A hit needs the correct movie
# and the field that answers the question, not just a matching title.
LOOKUPS = (
    ("How long does The Dark Knight run?", "tt0468569", "performance"),
    ("When was Inception released?", "tt1375666", "identity"),
    ("How many people voted for Interstellar on IMDb?", "tt0816692", "performance"),
    ("What genres is Mirror Game tagged with?", "tt5354160", "identity"),
    ("What is Unpaarvayil's IMDb score out of ten?", "tt15176374", "performance"),
    ("What year did The Fellowship of the Ring come out?", "tt0120737", "identity"),
    ("How many minutes is The Return of the King?", "tt0167260", "performance"),
    ("Which genre tags describe A Life on Our Planet?", "tt11989890", "identity"),
    ("What was the IMDb vote count for The Last Attack?", "tt33175825", "performance"),
    ("When was the film #Home released?", "tt10534500", "identity"),
)


def split_by_fields(chunks):
    for chunk in chunks:
        lines = chunk["text"].splitlines()
        title = f"Movie title: {chunk['source_title']}"
        identity = [line for line in lines if line.startswith(
            ("Release year:", "Genres:", "IMDb ID:")
        )]
        performance = [line for line in lines if line.startswith(
            ("Runtime (minutes):", "IMDb rating", "IMDb vote count:", "Rank in this export:")
        )]
        yield chunk["chunk_id"], "identity", "\n".join([title, *identity])
        yield chunk["chunk_id"], "performance", "\n".join([title, *performance])


def compare_chunks(chunks, encode):
    catalog = {chunk["chunk_id"] for chunk in chunks}
    if any(movie_id not in catalog for _, movie_id, _ in LOOKUPS):
        raise ValueError("The comparison expects the IMDb export included in this project.")

    variants = {
        "One chunk per movie": [(c["chunk_id"], "all", c["text"]) for c in chunks],
        "Two chunks per movie": list(split_by_fields(chunks)),
    }
    queries = encode([question for question, _, _ in LOOKUPS])
    results = []
    for name, records in variants.items():
        vectors = encode([text for _, _, text in records])
        top_three = np.argsort(-(queries @ vectors.T), axis=1, kind="stable")[:, :3]
        ranks = []
        for indices, (_, movie_id, required_group) in zip(top_three, LOOKUPS, strict=True):
            ranks.append(next(
                (rank for rank, i in enumerate(indices, start=1)
                 if records[i][0] == movie_id and records[i][1] in {"all", required_group}),
                0,
            ))
        results.append({
            "layout": name,
            "chunks": len(records),
            "hit_at_1": ranks.count(1),
            "hit_at_3": sum(rank > 0 for rank in ranks),
            "mrr_at_3": sum(1 / rank for rank in ranks if rank) / len(ranks),
            "misses": [question for (question, _, _), rank in zip(LOOKUPS, ranks, strict=True)
                       if not rank],
        })
    return results


def comparison_summary(results):
    first, second = results
    winner = max(results, key=lambda row: (row["hit_at_3"], row["mrr_at_3"], -row["chunks"]))
    if (first["hit_at_3"], first["mrr_at_3"]) == (second["hit_at_3"], second["mrr_at_3"]):
        return "The layouts tied on these questions. One chunk per movie uses half as many vectors."
    return (
        f"{winner['layout']} performed better on these ten questions. "
        "The comparison prioritizes finding the needed evidence in the first three results, "
        "then gives credit for finding it higher in the list."
    )


def main():
    parser = argparse.ArgumentParser(description="Compare two ways of chunking the IMDb export.")
    parser.add_argument("--chunks", type=Path, default=Path(__file__).with_name("imdb_chunks.jsonl"))
    parser.add_argument("--encoder", choices=("minilm", "tfidf"), default="minilm")
    args = parser.parse_args()
    chunks = load_chunks(args.chunks)
    if args.encoder == "minilm":
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(EMBEDDING_MODEL)

        def encode(texts):
            return embed(model, texts)
    else:
        from sklearn.feature_extraction.text import TfidfVectorizer

        vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        vectorizer.fit([chunk["text"] for chunk in chunks])

        def encode(texts):
            return vectorizer.transform(texts).toarray()

    results = compare_chunks(chunks, encode)
    print(f"Encoder: {args.encoder}. Fixed test set: {len(LOOKUPS)} questions.")
    print("| Layout | Chunks | Hit@1 | Hit@3 | MRR@3 |")
    print("|---|---:|---:|---:|---:|")
    for row in results:
        print(f"| {row['layout']} | {row['chunks']} | {row['hit_at_1']}/{len(LOOKUPS)} "
              f"| {row['hit_at_3']}/{len(LOOKUPS)} | {row['mrr_at_3']:.3f} |")
    print(comparison_summary(results))
    for row in results:
        if row["misses"]:
            print(f"Misses for {row['layout']}: " + " • ".join(row["misses"]))


if __name__ == "__main__":
    main()
