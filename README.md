# IMDb Movie Chat

Ask questions about the 20 movies in the Challenge 1 IMDb export. The export records IMDb IDs, titles, release years, genres, runtimes, ratings, votes, and ranks. Ranks refer to this export.

## Prepare the local dataset

Run the following once from the repository folder. The extractor downloads the official IMDb files and generates the ranked CSV locally. Ratings, vote counts, and the available titles may change in a fresh IMDb snapshot. The verified demo records the snapshot used for the live run.

```bash
python -m pip install -r requirements-scraper.txt
python part_1_extract.py --output imdb_top_movies_scrapy.csv
python part_2_prepare.py --csv imdb_top_movies_scrapy.csv --output imdb_chunks.jsonl
```

If you already have the Challenge 1 CSV, put your local copy at `imdb_top_movies_scrapy.csv` and run the preparation command. Keep the generated data and indexes local.

Information courtesy of [IMDb](https://www.imdb.com). Used with permission.

## Install and run

Run the following from this folder in a Python environment:

```bash
python -m pip install -r requirements.txt
python imdb_chatbot.py index
python -m streamlit run app.py
```

The first index build downloads the embedding model. Open the local URL printed by Streamlit, then enter your Gemini API key in the sidebar password field or set `GEMINI_API_KEY` before starting the app. You can get a key from [Google AI Studio](https://aistudio.google.com/apikey). Keep the key out of source control.

Streamlit defaults to `gemini-3.5-flash-lite`, the model used for the live demo. Its model selector also offers `gemini-2.5-flash-lite` from the challenge PDF. The tested project returned a 404 for 2.5 Flash-Lite, which [Google restricts to users with prior access](https://aistudio.google.com/docs/deprecations#gemini-2.5-flash-models).

For the terminal interface, use `python imdb_chatbot.py chat --model gemini-3.5-flash-lite` and type `exit` when finished. Omit the model flag if your project can use 2.5 Flash-Lite.

## Run in Colab

Extract the project files into `/content/imdb-rag`. Prepare the local dataset from that folder using the commands above before building the index. Run each line below in a notebook cell:

```python
%cd /content/imdb-rag
%pip -q install -r requirements.txt
%run imdb_chatbot.py index
%run imdb_chatbot.py chat --model gemini-3.5-flash-lite
```

## Data and citations

The program builds its FAISS index once from `imdb_chunks.jsonl`, which contains one record per movie in `imdb_top_movies_scrapy.csv`. The data comes from the Challenge 1 extraction of the [IMDb datasets](https://datasets.imdbws.com/).

At question time, FAISS retrieves the closest movie chunks. Gemini proposes a structured query plan using the title catalog and retrieval hints. Python applies the plan to the export and calculates each result. Gemini then receives the retrieved chunks, any additional records needed to support the calculation, and the verified result to produce the final reply. Python checks that the reply preserves every verified line and attaches citations from the records used. If the generated wording fails that check, the chat labels and displays the verified calculation. The API calls use no web search. Accuracy still depends on how Gemini interprets the question. Missing fields and unsupported requests are declined.

See [DEMO.md](DEMO.md) for the live results and the refusal checks.

## Optional challenge features

- **Similarity scores.** The terminal shows the three closest chunks and their cosine scores. In Streamlit, expand **Closest retrieved chunks** under the answer. The scores measure retrieval similarity. Citations show records used to support the answer, including any additional records used for calculations.
- **Follow-up questions.** The last three exchanges help Gemini resolve references such as “it.” Python still calculates each answer from the export. Streamlit keeps the displayed conversation in the current session and has a **Clear conversation** button.
- **Chunking comparison.** Open the **Chunking comparison** tab or run `python benchmark_chunks.py` with the chatbot's MiniLM model. For an offline word-based comparison, run `python benchmark_chunks.py --encoder tfidf`. Both commands run without a Gemini API key. Use `%run benchmark_chunks.py` in Colab.

### Chunking results

The comparison uses the original demo export and runs ten fixed title-and-field lookups on two layouts. If a fresh export lacks a tested title, use the original local snapshot for the benchmark. One layout keeps all fields for each movie together. The other repeats the title in two chunks, with release year and genres in one, and runtime, rating, votes, and rank in the other. A hit requires both the correct movie and the field needed for the question. Hit@1 and Hit@3 count how often that evidence appears first or in the first three results. MRR@3 rewards earlier hits.

| Layout | Chunks | TF-IDF Hit@1 | TF-IDF Hit@3 | TF-IDF MRR@3 |
|---|---:|---:|---:|---:|
| One chunk per movie | 20 | 10/10 | 10/10 | 1.000 |
| Two chunks per movie | 40 | 6/10 | 10/10 | 0.800 |

For this export and TF-IDF test set, a single chunk per movie is the better default. It showed the right field first more often and used half as many vectors. The MiniLM comparison reports its own results when run.
