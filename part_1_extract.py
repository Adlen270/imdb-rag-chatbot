import argparse
import csv
import gzip
import json
import subprocess
import sys
from html import escape
from pathlib import Path
from tempfile import TemporaryDirectory

from part_2_prepare import DATA_DIR

BASE_URL = "https://datasets.imdbws.com/"
BASICS_FILE = "title.basics.tsv.gz"
RATINGS_FILE = "title.ratings.tsv.gz"
MISSING = r"\N"
FIRST_YEAR = 2000
LAST_YEAR = 2025
MIN_VOTES = 10_000
TOP_N = 20
HTML_PAGE_SIZE = 5

BASICS_FIELDS = (
    "tconst", "titleType", "primaryTitle", "originalTitle", "isAdult",
    "startYear", "endYear", "runtimeMinutes", "genres",
)
RATINGS_FIELDS = ("tconst", "averageRating", "numVotes")
OUTPUT_FIELDS = (
    "rank", "imdb_id", "title", "year", "genres", "runtime_minutes",
    "imdb_rating", "votes",
)
NUMERIC_FIELDS = {
    "rank": int,
    "year": int,
    "runtime_minutes": int,
    "imdb_rating": float,
    "votes": int,
}
ATTRIBUTION = "Information courtesy of\nIMDb\n(https://www.imdb.com).\nUsed with permission."


def download_if_missing(session, filename, data_dir):
    destination = data_dir / filename
    if destination.is_file():
        return destination

    with session.get(BASE_URL + filename, stream=True, timeout=(15, 120)) as response:
        response.raise_for_status()
        with TemporaryDirectory(dir=data_dir) as directory:
            temporary_path = Path(directory) / filename
            with temporary_path.open("wb") as partial:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    partial.write(chunk)
            with temporary_path.open("rb") as partial:
                if partial.read(2) != b"\x1f\x8b":
                    raise ValueError(f"IMDb did not return a gzip archive for {filename}.")
            temporary_path.replace(destination)
    return destination


def read_tsv(path, required_columns):
    with gzip.open(path, mode="rt", encoding="utf-8", newline="") as source:
        rows = csv.DictReader(source, delimiter="\t", quoting=csv.QUOTE_NONE)
        missing_columns = set(required_columns) - set(rows.fieldnames or [])
        if missing_columns:
            raise ValueError(f"{path.name} lacks columns: {', '.join(sorted(missing_columns))}")
        yield from rows


def load_ratings(path):
    ratings = {}
    for row in read_tsv(path, RATINGS_FIELDS):
        votes = int(row["numVotes"])
        if votes >= MIN_VOTES:
            ratings[row["tconst"]] = (float(row["averageRating"]), votes)
    return ratings


def select_movies(basics_path, ratings):
    movies = []
    for row in read_tsv(basics_path, BASICS_FIELDS):
        if row["titleType"] != "movie" or row["isAdult"] != "0":
            continue
        if row["startYear"] == MISSING:
            continue
        year = int(row["startYear"])
        if not FIRST_YEAR <= year <= LAST_YEAR:
            continue

        imdb_id = row["tconst"]
        rating = ratings.get(imdb_id)
        if rating is None:
            continue
        score, votes = rating
        movies.append({
            "imdb_id": imdb_id,
            "title": row["primaryTitle"],
            "year": year,
            "genres": None if row["genres"] == MISSING else row["genres"],
            "runtime_minutes": None if row["runtimeMinutes"] == MISSING else int(row["runtimeMinutes"]),
            "imdb_rating": score,
            "votes": votes,
        })
    return movies


def rank_movies(movies):
    movies.sort(key=lambda movie: (-movie["imdb_rating"], -movie["votes"], movie["imdb_id"]))
    return [{"rank": rank, **movie} for rank, movie in enumerate(movies[:TOP_N], start=1)]


def write_html_pages(rows, directory):
    page_count = max(1, (len(rows) + HTML_PAGE_SIZE - 1) // HTML_PAGE_SIZE)
    headings = "".join(f"<th>{field}</th>" for field in OUTPUT_FIELDS)

    for page_number in range(1, page_count + 1):
        offset = (page_number - 1) * HTML_PAGE_SIZE
        table_rows = []
        for row in rows[offset:offset + HTML_PAGE_SIZE]:
            cells = []
            for field in OUTPUT_FIELDS:
                value = "" if row[field] is None else str(row[field])
                cells.append(f"<td>{escape(value)}</td>")
            table_rows.append("<tr>" + "".join(cells) + "</tr>")

        next_link = ""
        if page_number < page_count:
            next_link = f'<a rel="next" href="page-{page_number + 1}.html">Next page</a>'
        html = (
            '<!doctype html><html lang="en"><head><meta charset="utf-8">'
            "<title>IMDb ranked report</title></head><body>"
            f"<table><thead><tr>{headings}</tr></thead>"
            f"<tbody>{''.join(table_rows)}</tbody></table>{next_link}</body></html>"
        )
        (directory / f"page-{page_number}.html").write_text(html, encoding="utf-8")
    return directory / "page-1.html", page_count


SCRAPY_WORKER = r'''
import json
import sys

from bs4 import BeautifulSoup
from scrapy import Spider, signals
from scrapy.crawler import CrawlerProcess


class LocalReportSpider(Spider):

    name = "imdb_local_report"

    def __init__(self, first_page, **kwargs):
        super().__init__(**kwargs)
        self.start_urls = [first_page]

    def parse(self, response):
        soup = BeautifulSoup(response.body, "html.parser", from_encoding="utf-8")
        for table_row in soup.select("tbody tr"):
            cells = [cell.get_text() for cell in table_row.find_all("td", recursive=False)]
            yield {"cells": cells}

        next_link = soup.select_one('a[rel="next"]')
        if next_link:
            yield response.follow(next_link["href"], callback=self.parse)


def crawl(first_page):
    rows = []
    errors = []

    def collect_item(item):
        rows.append(item["cells"])

    def collect_error(failure):
        errors.append(failure.value)

    process = CrawlerProcess(settings={
        "LOG_LEVEL": "ERROR",
        "TELNETCONSOLE_ENABLED": False,
        "DOWNLOAD_HANDLERS": {"http": None, "https": None},
    })
    crawler = process.create_crawler(LocalReportSpider)
    crawler.signals.connect(collect_item, signal=signals.item_scraped, weak=False)
    crawler.signals.connect(collect_error, signal=signals.spider_error, weak=False)
    process.crawl(crawler, first_page=first_page)
    process.start()

    if errors:
        raise ValueError(f"Could not read the local HTML report: {errors[0]}") from errors[0]
    return {"pages": crawler.stats.get_value("response_received_count", 0), "rows": rows}


print(json.dumps(crawl(sys.argv[1])))
'''


def crawl_local_report(first_page, expected_pages):
    result = subprocess.run(
        [sys.executable, "-c", SCRAPY_WORKER, first_page.resolve().as_uri()],
        capture_output=True,
        encoding="utf-8",
    )
    if result.returncode:
        details = result.stderr.strip() or result.stdout.strip()
        raise ValueError(f"Scrapy process failed (exit {result.returncode}).\n{details}")

    report = json.loads(result.stdout)
    if report["pages"] != expected_pages:
        raise ValueError("Scrapy did not read every local HTML page.")

    rows = []
    for cells in report["rows"]:
        row = dict(zip(OUTPUT_FIELDS, cells, strict=True))
        for field, convert in NUMERIC_FIELDS.items():
            row[field] = convert(row[field]) if row[field] else None
        row["genres"] = row["genres"] or None
        rows.append(row)
    return sorted(rows, key=lambda row: row["rank"])


def write_report(rows, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def show_results(rows):
    if not rows:
        print("No movies matched the filters.")
        return

    headings = ("Rank", "IMDb ID", "Title", "Year", "Genres", "Runtime (min)", "Rating", "Votes")
    table_rows = [headings]
    for row in rows:
        runtime = row["runtime_minutes"]
        table_rows.append((
            str(row["rank"]), row["imdb_id"], row["title"], str(row["year"]),
            row["genres"] or "-", "-" if runtime is None else str(runtime),
            f'{row["imdb_rating"]:.1f}', f'{row["votes"]:,}',
        ))

    widths = [max(map(len, column)) for column in zip(*table_rows)]
    for index, row in enumerate(table_rows):
        print(" | ".join(value.ljust(width) for value, width in zip(row, widths)).rstrip())
        if index == 0:
            print("-+-".join("-" * width for width in widths))


def run_report(data_dir, output_path):
    import requests

    data_dir.mkdir(parents=True, exist_ok=True)
    with requests.Session() as session:
        ratings_path = download_if_missing(session, RATINGS_FILE, data_dir)
        basics_path = download_if_missing(session, BASICS_FILE, data_dir)

    ratings = load_ratings(ratings_path)
    movies = select_movies(basics_path, ratings)
    ranked = rank_movies(movies)

    with TemporaryDirectory(prefix="imdb-html-") as directory:
        first_page, page_count = write_html_pages(ranked, Path(directory))
        extracted = crawl_local_report(first_page, page_count)

    if extracted != ranked:
        raise ValueError("The HTML extraction differs from the ranking calculated from the TSV files.")

    write_report(extracted, output_path)
    return extracted


def write_demo_files(data_dir):
    titles = [
        ("tt9900001", "movie", "Glass Harbor", "Glass Harbor", "0", "2019", MISSING, "102", "Drama,Mystery"),
        ("tt9900002", "movie", "Blue Hour", "Blue Hour", "0", "2023", MISSING, "98", "Drama"),
        ("tt9900003", "movie", "Northbound Light", "Northbound Light", "0", "2016", MISSING, MISSING, "Adventure"),
        ("tt9900004", "short", "Night Signal", "Night Signal", "0", "2022", MISSING, "15", "Short"),
        ("tt9900005", "movie", "Quiet Street", "Quiet Street", "0", "2020", MISSING, "91", "Drama"),
        ("tt9900006", "movie", "Old Harbour", "Old Harbour", "0", "1998", MISSING, "89", "Drama"),
        ("tt9900007", "movie", "Unnamed Year", "Unnamed Year", "0", MISSING, MISSING, "92", "Drama"),
        ("tt9900008", "movie", "Unrated Story", "Unrated Story", "0", "2021", MISSING, "96", "Drama"),
    ]
    ratings = [
        ("tt9900001", "8.4", "18000"),
        ("tt9900002", "8.4", "24000"),
        ("tt9900003", "7.9", "14250"),
        ("tt9900004", "9.2", "25000"),
        ("tt9900005", "9.0", "9999"),
        ("tt9900006", "9.1", "50000"),
        ("tt9900007", "8.9", "12000"),
    ]
    for filename, header, records in (
        (BASICS_FILE, BASICS_FIELDS, titles),
        (RATINGS_FILE, RATINGS_FIELDS, ratings),
    ):
        with gzip.open(data_dir / filename, mode="wt", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
            writer.writerow(header)
            writer.writerows(records)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build a top-20 IMDb report with a local HTML crawl.")
    parser.add_argument("--demo", action="store_true", help="Use fictional data without a network connection.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR / "imdb_downloads", help="Official download cache.")
    parser.add_argument("--output", type=Path, help="Path for the output CSV.")
    args = parser.parse_args(argv)

    try:
        if args.demo:
            with TemporaryDirectory(prefix="imdb-demo-") as directory:
                data_dir = Path(directory)
                write_demo_files(data_dir)
                rows = run_report(data_dir, args.output or DATA_DIR / "imdb_demo.csv")
            print("Demo results (fictional data)")
            show_results(rows)
        else:
            rows = run_report(args.data_dir, args.output or DATA_DIR / "imdb_latest.csv")
            show_results(rows)
            print("\n" + ATTRIBUTION)
    except (OSError, EOFError, ValueError, csv.Error) as error:
        parser.exit(1, f"Extraction failed: {error}\n")


if __name__ == "__main__":
    main()
