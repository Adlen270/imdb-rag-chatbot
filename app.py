import os
from itertools import groupby
from pathlib import Path

import streamlit as st
from google import genai
from google.genai import errors

from benchmark_chunks import LOOKUPS, compare_chunks, comparison_summary
from imdb_chatbot import (
    DEFAULT_GEMINI_MODEL,
    answer_question,
    conversation_turn,
    embed,
    load_chat_resources,
    retrieve_matches,
)


PROJECT = Path(__file__).resolve().parent
DEMO_MODEL = "gemini-3.5-flash-lite"


@st.cache_resource
def resources():
    return load_chat_resources(PROJECT / "imdb_chunks.jsonl", PROJECT / "imdb_movies.faiss")


@st.cache_data
def comparison():
    chunks, _, model = resources()
    return compare_chunks(chunks, lambda texts: embed(model, texts))


def answer_markdown(answer):
    blocks = []
    for is_list, lines in groupby(answer.splitlines(), lambda line: line.startswith("- ")):
        if is_list:
            blocks.append("\n".join(lines))
        else:
            blocks.extend(line for line in lines if line.strip())
    return "\n\n".join(blocks)


def render_reply(turn, show_scores):
    st.markdown(answer_markdown(turn["answer"]))
    if turn["sources"]:
        with st.expander(f"IMDb sources · {len(turn['sources'])}"):
            for source in turn["sources"]:
                st.markdown(f"- [{source['source_title']}]({source['source_url']})")
    if show_scores:
        with st.expander("Closest retrieved chunks"):
            st.table([
                {"Movie": chunk["source_title"], "Cosine similarity": f"{score:.3f}"}
                for chunk, score in turn["matches"]
            ])
            st.caption("Similarity measures the query match. The answer can cite additional records used in a calculation.")


def show_chat(api_key, gemini_model, show_scores):
    turns = st.session_state.turns
    if not turns:
        st.markdown("Ask which movie outside the top 10 received the most votes, "
                    "then follow up with **What was its IMDb rating?**")
    for turn in turns:
        with st.chat_message("user"):
            st.text(turn["question"])
        with st.chat_message("assistant"):
            render_reply(turn, show_scores)

    if not api_key:
        st.info("Enter your Gemini API key in the sidebar to start the chat.")
    question = st.chat_input("Ask about the IMDb export", disabled=not api_key)
    if not question:
        return
    with st.chat_message("user"):
        st.text(question)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Retrieving records and checking the answer…"):
                chunks, index, model = resources()
                matches = retrieve_matches(question, chunks, index, model)
                with genai.Client(api_key=api_key) as client:
                    answer, sources = answer_question(
                        question, chunks, index, model, client, gemini_model,
                        history=[turn["memory"] for turn in turns[-3:]], matches=matches,
                    )
        except errors.APIError as error:
            st.error(f"Gemini returned an API error ({error.code}). Check model access and quota in Google AI Studio.")
            return
        except (OSError, RuntimeError, ValueError) as error:
            st.error(str(error))
            return

        turn = {
            "question": question, "answer": answer, "sources": sources, "matches": matches,
            "memory": conversation_turn(question, answer, sources),
        }
        turns.append(turn)
        render_reply(turn, show_scores)


def show_comparison():
    st.subheader("Which chunk layout retrieves better evidence?")
    st.markdown("The comparison uses the same embedding model and ten fixed questions. "
                "A hit requires both the right movie and the field needed to answer.")
    st.markdown("**One chunk per movie** keeps all fields together. **Two chunks per movie** "
                "separates identity and genres from ratings, votes, and runtime.")
    if st.button("Run retrieval comparison", type="primary"):
        try:
            with st.spinner("Embedding both layouts and scoring the questions…"):
                st.session_state.chunk_comparison = comparison()
        except (OSError, RuntimeError, ValueError) as error:
            st.error(str(error))
    results = st.session_state.get("chunk_comparison")
    if results:
        st.table([
            {"Layout": row["layout"], "Chunks": row["chunks"],
             "Hit@1": f"{row['hit_at_1']}/{len(LOOKUPS)}",
             "Hit@3": f"{row['hit_at_3']}/{len(LOOKUPS)}",
             "MRR@3": f"{row['mrr_at_3']:.3f}"}
            for row in results
        ])
        st.write(comparison_summary(results))
        for row in results:
            if row["misses"]:
                with st.expander(f"Top-three misses · {row['layout']}"):
                    for question in row["misses"]:
                        st.text(question)


def main():
    st.set_page_config(page_title="IMDb Movie Chat", layout="centered")
    if "turns" not in st.session_state:
        st.session_state.turns = []
    st.title("IMDb Movie Chat")
    st.caption("Search and compare the 20 titles in your scraped IMDb export.")

    with st.sidebar:
        st.header("Chat settings")
        gemini_model = st.selectbox("Gemini model", [DEMO_MODEL, DEFAULT_GEMINI_MODEL])
        api_key = os.getenv("GEMINI_API_KEY") or st.text_input("Gemini API key", type="password").strip()
        show_scores = st.toggle("Show retrieval scores", value=True)
        if st.button("Clear conversation"):
            st.session_state.turns = []
            st.rerun()

    chat, evaluation = st.tabs(["Chat", "Chunking comparison"])
    with chat:
        show_chat(api_key, gemini_model, show_scores)
    with evaluation:
        show_comparison()


if __name__ == "__main__":
    main()
