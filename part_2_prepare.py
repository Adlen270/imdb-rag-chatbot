import argparse
import csv
import json
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from tempfile import TemporaryDirectory


DATA_DIR = Path(__file__).resolve().parent / "data"
CSV_PATH = DATA_DIR / "imdb_top_movies_scrapy.csv"
CHUNKS_PATH = DATA_DIR / "imdb_chunks.jsonl"
CHUNK_LABELS = {
    "rank": "Rank in this export",
    "year": "Release year",
    "genres": "Genres",
    "runtime": "Runtime (minutes)",
    "rating": "IMDb rating (out of 10)",
    "votes": "IMDb vote count",
}
NUMERIC_PARSERS = {
    "rank": int, "year": int, "runtime": int, "rating": Decimal, "votes": int,
}
CSV_LABELS = {
    "title": "Movie title",
    "year": CHUNK_LABELS["year"],
    "genres": CHUNK_LABELS["genres"],
    "runtime_minutes": CHUNK_LABELS["runtime"],
    "imdb_rating": CHUNK_LABELS["rating"],
    "votes": CHUNK_LABELS["votes"],
    "rank": CHUNK_LABELS["rank"],
    "imdb_id": "IMDb ID",
}


def movie_values(chunk):
    labels = dict(line.split(": ", 1) for line in chunk["text"].splitlines())
    values = {"title": chunk["source_title"], "imdb_id": chunk["chunk_id"]}
    for field, label in CHUNK_LABELS.items():
        value = labels[label].strip()
        if not value or value == r"\N":
            values[field] = None
        elif field == "genres":
            values[field] = [genre.strip() for genre in value.split(",")]
        else:
            values[field] = NUMERIC_PARSERS[field](value)
    return values


def chunks_with_values(chunks):
    return [(chunk, movie_values(chunk)) for chunk in chunks]


def validate_chunks(chunks):
    if not chunks:
        raise ValueError("The export contains no movies.")
    seen_ids = set()
    seen_ranks = set()
    for chunk, movie in chunks_with_values(chunks):
        movie_id = movie["imdb_id"]
        rank, year, rating, votes = (movie[field] for field in ("rank", "year", "rating", "votes"))
        runtime = movie["runtime"]
        if not re.fullmatch(r"tt[0-9]+", movie_id) or not movie["title"]:
            raise ValueError(f"Invalid ID or empty title: {movie_id!r}.")
        if chunk["source_url"] != f"https://www.imdb.com/title/{movie_id}/":
            raise ValueError(f"Invalid citation URL for {movie_id}.")
        if rank is None or rank < 1 or year is None or year < 1 or votes is None or votes < 0:
            raise ValueError(f"Invalid rank, release year, or vote count for {movie_id}.")
        if rating is None or not rating.is_finite() or not 0 <= rating <= 10:
            raise ValueError(f"Invalid IMDb rating for {movie_id}.")
        if runtime is not None and runtime <= 0:
            raise ValueError(f"Invalid runtime for {movie_id}.")
        if movie_id in seen_ids or rank in seen_ranks:
            raise ValueError(f"Duplicate IMDb ID or export rank: {movie_id}, rank {rank}.")
        seen_ids.add(movie_id)
        seen_ranks.add(rank)


def chunks_from_csv(csv_path):
    chunks = []
    with csv_path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = set(CSV_LABELS) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Missing CSV columns: {', '.join(sorted(missing))}")
        for row in reader:
            values = {field: " ".join((row[field] or "").split()) for field in CSV_LABELS}
            movie_id = values["imdb_id"]
            chunks.append({
                "chunk_id": movie_id,
                "text": "\n".join(f"{label}: {values[field]}" for field, label in CSV_LABELS.items()),
                "source_title": values["title"],
                "source_url": f"https://www.imdb.com/title/{movie_id}/",
            })
    validate_chunks(chunks)
    return chunks


def load_chunks(path=CHUNKS_PATH):
    if not path.is_file():
        raise ValueError(f"Chunks file not found: {path}. Run python part_2_prepare.py first.")
    with path.open(encoding="utf-8") as stream:
        chunks = [json.loads(line) for line in stream if line.strip()]
    validate_chunks(chunks)
    return chunks


def write_chunks(chunks, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as directory:
        staged = Path(directory) / output.name
        with staged.open("w", encoding="utf-8", newline="\n") as stream:
            for chunk in chunks:
                stream.write(json.dumps(chunk, ensure_ascii=False) + "\n")
        staged.replace(output)


def main():
    parser = argparse.ArgumentParser(description="Convert the IMDb CSV into one chunk per movie.")
    parser.add_argument("--csv", type=Path, default=CSV_PATH)
    parser.add_argument("--output", type=Path, default=CHUNKS_PATH)
    args = parser.parse_args()
    try:
        chunks = chunks_from_csv(args.csv)
        write_chunks(chunks, args.output)
    except (OSError, ValueError, InvalidOperation) as error:
        parser.exit(1, f"Could not prepare chunks: {error}\n")
    print(f"Wrote {len(chunks)} movie chunks to {args.output}")


if __name__ == "__main__":
    main()
