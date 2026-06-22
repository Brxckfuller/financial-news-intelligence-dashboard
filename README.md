# Financial News Intelligence Dashboard

![Dashboard Overview](screenshots/dashboard_overview.png)

This project was built to explore how modern NLP techniques can be applied to financial news monitoring.

The system automatically collects headlines from multiple financial news sources, analyses sentiment using FinBERT, generates semantic embeddings with Sentence Transformers, and surfaces emerging themes through clustering and interactive visualisations.

The project was developed as a practical exercise in building an end-to-end AI pipeline rather than training a custom model. The focus was on data ingestion, NLP workflows, semantic search, automation and dashboard development.


## How to Run

### Step 1: Clone the repository

```bash
git clone https://github.com/Brxckfuller/financial-news-intelligence-dashboard.git
cd financial-news-intelligence-dashboard
```

### Step 2: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Collect the latest financial news

```bash
python app/ingest.py
```

### Step 4: Run sentiment analysis

```bash
python app/analyse.py
```

### Step 5: Generate semantic embeddings

```bash
python app/semantic_search.py
```

### Step 6: Launch the dashboard

```bash
streamlit run app/dashboard.py
```

### Step 7: Open the dashboard

Navigate to:

```text
http://localhost:8501
```




## Technical Highlights

- Aggregates news from MarketWatch, CNBC, Yahoo Finance, Nasdaq and Investing.com RSS feeds
- Filters and deduplicates market-relevant headlines
- Applies FinBERT sentiment classification to generate bullish, bearish and neutral signals
- Generates semantic embeddings using all-MiniLM-L6-v2
- Implements semantic headline search and related-headline recommendations
- Uses K-Means clustering to identify emerging market themes
- Tracks sentiment history across pipeline runs
- Presents results through an interactive Streamlit dashboard

## Project Structure

app/
- ingest.py
- analyse.py
- semantic_search.py
- dashboard.py
- scheduled_pipeline.py

data/
- news.csv
- analysed_news.csv
- sentiment_history.csv
- headline_embeddings.npy

## Author

Brock Fuller

Master of Artificial Intelligence – RMIT University
