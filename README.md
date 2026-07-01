# Financial News Intelligence Dashboard

A financial news analytics dashboard built with Python, FinBERT, Sentence Transformers and Streamlit.

The project collects recent financial headlines from RSS feeds, analyses market sentiment, creates semantic embeddings, and displays the results in an interactive dashboard.

It was built to practise an end-to-end NLP workflow: data ingestion, cleaning, sentiment classification, semantic search, clustering and dashboard development.


---

## What it does

- Collects financial headlines from multiple RSS feeds
- Cleans and deduplicates market news
- Uses FinBERT to classify headlines as bullish, bearish or neutral
- Generates semantic embeddings using Sentence Transformers
- Supports semantic headline search
- Groups related headlines into market themes
- Tracks sentiment history across pipeline runs
- Presents results in a Streamlit dashboard

---

## Pipeline

```text
RSS Feeds
   │
   ▼
Headline Collection
   │
   ▼
Cleaning and Deduplication
   │
   ├───────────────┐
   ▼               ▼
FinBERT       Sentence Transformers
   │               │
   ▼               ▼
Sentiment      Embeddings
   │               │
   └───────┬───────┘
           ▼
     Topic Clustering
           │
           ▼
   Streamlit Dashboard
```

---

## Models used

- **FinBERT** for financial sentiment classification
- **Sentence Transformers** for semantic headline embeddings
- **K-Means** for clustering related market themes

---

## Tech stack

- Python
- Pandas
- Streamlit
- FinBERT
- Sentence Transformers
- scikit-learn
- RSS feed parsing

---

## How to run

Clone the repository:

```bash
git clone https://github.com/Brxckfuller/financial-news-intelligence-dashboard.git
cd financial-news-intelligence-dashboard
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Collect the latest financial news:

```bash
python app/ingest.py
```

Run sentiment analysis:

```bash
python app/analyse.py
```

Generate semantic embeddings:

```bash
python app/semantic_search.py
```

Launch the dashboard:

```bash
streamlit run app/dashboard.py
```

The app will open at:

```text
http://localhost:8501
```

---

## Project structure

```text
financial-news-intelligence-dashboard/

├── app/
│   ├── ingest.py
│   ├── analyse.py
│   ├── semantic_search.py
│   ├── scheduled_pipeline.py
│   └── dashboard.py
│
├── data/
│   ├── news.csv
│   ├── analysed_news.csv
│   ├── sentiment_history.csv
│   └── headline_embeddings.npy
│
├── screenshots/
│   └── dashboard_overview.png
│
├── requirements.txt
└── README.md
```

---

## Example use case

A user can search for a market theme such as:

```text
AI stocks
```

or

```text
interest rates
```

The dashboard returns semantically related headlines rather than relying only on exact keyword matches.

---

