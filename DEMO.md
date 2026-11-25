# Challenge 2 demo

All five answers below passed live in Colab with the updated answer-generation step and Gemini 3.5 Flash-Lite. Each reply preserved the verified result and showed its IMDb sources. None used the local fallback. The answers were checked against the 20-row IMDb export.

The same Google project returned a 404 when asked to use the required Gemini 2.5 Flash-Lite model. The live results below therefore apply to 3.5 Flash-Lite. Organizer approval for the substitution is pending.

| Question | Live result | IMDb sources |
|---|---|---|
| Find movies tagged both Adventure and Fantasy. List them chronologically and calculate their total runtime and average IMDb rating. | The Fellowship of the Ring (2001, 178 min, 8.9), The Two Towers (2002, 179 min, 8.8), and The Return of the King (2003, 201 min, 9.0). **558 minutes** in total and **8.9/10** on average. | [Fellowship](https://www.imdb.com/title/tt0120737/), [Two Towers](https://www.imdb.com/title/tt0167261/), [Return of the King](https://www.imdb.com/title/tt0167260/) |
| Among movies rated exactly 8.7, name the two with the most votes and calculate the gap. | **Interstellar** (2,638,552) and **Kill Bill: The Whole Bloody Affair** (71,117). The gap is **2,567,435 votes**. | [Interstellar](https://www.imdb.com/title/tt0816692/), [Kill Bill](https://www.imdb.com/title/tt6019206/) |
| Which release year appears most often? Which movie from that year has the highest rating, with ties broken by votes? | **2021** has **3 movies**. Their ratings are tied at 8.7. **#Home** wins with **20,400 votes**. | [#Home](https://www.imdb.com/title/tt10534500/), [Kadaisi Vivasayi](https://www.imdb.com/title/tt10300570/), [The Silence of Swastika](https://www.imdb.com/title/tt16747572/) |
| Among thrillers released in 2010 or later and shorter than 150 minutes, list the top three by rating, breaking ties by votes. | **Unpaarvayil** (9.0, 117 min), **Inception** (8.8, 148 min), **Mirror Game** (8.8, 147 min). | [Unpaarvayil](https://www.imdb.com/title/tt15176374/), [Inception](https://www.imdb.com/title/tt1375666/), [Mirror Game](https://www.imdb.com/title/tt5354160/) |
| Who has the most votes among ranks 11–20? Which top-10 movies have more votes? | **Interstellar**, rank 13, with **2,638,552** votes. **The Dark Knight** (3,253,799) and **Inception** (2,892,755) exceed it in the top 10. | [Interstellar](https://www.imdb.com/title/tt0816692/), [The Dark Knight](https://www.imdb.com/title/tt0468569/), [Inception](https://www.imdb.com/title/tt1375666/) |

## Refusal checks

Both cases passed live without citations on the updated chatbot, using the Gemini 3.5 Flash-Lite Colab session described above.

1. **Who directed The Dark Knight?** The movie is in the export, but director is outside the field list.
2. **What is the IMDb rating of Titanic (1997)?** The film is outside the 20-movie export.

Observed reply for both was “I couldn't find that information in the IMDb data provided.”

## Optional feature demo

Ask **Which movie outside the top 10 received the most votes?** The expected answer is **Interstellar**, with **2,638,552** votes. Then ask **What was its IMDb rating?** The expected answer is **8.7/10**, citing the [same movie](https://www.imdb.com/title/tt0816692/). Both turns show similarity scores in the terminal and under **Closest retrieved chunks** in Streamlit.

The Streamlit app passed a live check in Colab for the Adventure and Fantasy question above. It listed all three films, calculated a total of 558 minutes and an average rating of 8.9/10, and displayed three IMDb sources. The retrieval score panel was visible below the answer.

The follow-up logic and score display passed local integration checks with simulated Gemini replies. The live follow-up and MiniLM comparison have yet to be verified. The README records the measured TF-IDF comparison and shows how to run it with MiniLM.
