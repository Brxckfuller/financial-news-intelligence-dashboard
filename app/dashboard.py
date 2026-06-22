from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

st.set_page_config(
    page_title="Financial News Intelligence",
    page_icon="📈",
    layout="wide"
)

base_path = Path(__file__).parent.parent

data_path = base_path / "data" / "analysed_news.csv"
history_path = base_path / "data" / "sentiment_history.csv"
embeddings_path = base_path / "data" / "headline_embeddings.npy"

df = pd.read_csv(data_path)

st.title("Financial News Intelligence Dashboard")

if "timestamp" in df.columns:
    latest_time = pd.to_datetime(df["timestamp"]).max()
    st.caption(f"Last updated: {latest_time.strftime('%Y-%m-%d %H:%M:%S')}")

st.subheader("Dashboard Summary")

bullish_count = len(df[df["signal"] == "Bullish"])
bearish_count = len(df[df["signal"] == "Bearish"])
neutral_count = len(df[df["signal"] == "Neutral"])

col1, col2, col3, col4 = st.columns(4)

col1.metric("Headlines", len(df))
col2.metric("Bullish", bullish_count)
col3.metric("Bearish", bearish_count)
col4.metric("Neutral", neutral_count)

st.subheader("Semantic Headline Search")

query = st.text_input(
    "Search by meaning",
    placeholder="Examples: AI stocks, interest rates, oil risk, bitcoin weakness"
)

if embeddings_path.exists():
    headline_embeddings = np.load(embeddings_path)

    if len(headline_embeddings) != len(df):
        st.warning("Embeddings are out of sync. Run semantic_search.py again.")
        display_df = df

    elif query:
        model = SentenceTransformer("all-MiniLM-L6-v2")

        query_embedding = model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        similarities = headline_embeddings @ query_embedding.T
        similarities = similarities.flatten()

        display_df = df.copy()
        display_df["similarity"] = similarities

        display_df = (
            display_df
            .sort_values(by="similarity", ascending=False)
            .head(10)
        )

    else:
        display_df = df

else:
    st.warning("Embeddings file not found. Run semantic_search.py first.")
    headline_embeddings = None
    display_df = df

st.dataframe(
    display_df,
    use_container_width=True,
    height=350
)

st.subheader("Related Headline Suggestions")

if embeddings_path.exists() and headline_embeddings is not None and len(headline_embeddings) == len(df):
    selected_headline = st.selectbox(
        "Choose a headline to find similar headlines",
        df["headline"].tolist()
    )

    selected_index = df[df["headline"] == selected_headline].index[0]
    selected_embedding = headline_embeddings[selected_index]

    similarities = headline_embeddings @ selected_embedding

    related_df = df.copy()
    related_df["similarity"] = similarities

    related_df = (
        related_df
        .drop(index=selected_index)
        .sort_values(by="similarity", ascending=False)
        .head(5)
    )

    st.table(
        related_df[
            [
                col for col in [
                    "source",
                    "headline",
                    "ticker",
                    "signal",
                    "sentiment",
                    "similarity"
                ]
                if col in related_df.columns
            ]
        ]
    )

else:
    st.info("Run semantic_search.py to enable related headline suggestions.")

st.subheader("Emerging Topic Clusters")

if embeddings_path.exists() and headline_embeddings is not None and len(headline_embeddings) == len(df):
    number_of_clusters = min(5, len(df))

    kmeans = KMeans(
        n_clusters=number_of_clusters,
        random_state=42,
        n_init=10
    )

    clusters = kmeans.fit_predict(headline_embeddings)

    cluster_df = df.copy()
    cluster_df["cluster"] = clusters

    cluster_summary = (
        cluster_df
        .groupby("cluster")
        .agg(
            headline_count=("headline", "count"),
            average_sentiment=("sentiment", "mean")
        )
        .reset_index()
        .sort_values(by="headline_count", ascending=False)
    )

    st.dataframe(
        cluster_summary,
        use_container_width=True
    )

    cluster_chart = (
        alt.Chart(cluster_summary)
        .mark_bar()
        .encode(
            x=alt.X("cluster:N", title="Topic Cluster"),
            y=alt.Y("headline_count:Q", title="Headline Count"),
            color=alt.Color("average_sentiment:Q", title="Average Sentiment"),
            tooltip=[
                "cluster",
                "headline_count",
                "average_sentiment"
            ]
        )
        .properties(height=350)
    )

    st.altair_chart(
        cluster_chart,
        use_container_width=True
    )

    for cluster_id in cluster_summary["cluster"]:
        examples = (
            cluster_df[cluster_df["cluster"] == cluster_id]
            .head(5)
        )

        cluster_name = examples.iloc[0]["headline"][:50]

        with st.expander(f"🔥 {cluster_name}..."):
            st.table(
                examples[
                    [
                        col for col in [
                            "source",
                            "headline",
                            "ticker",
                            "signal",
                            "sentiment"
                        ]
                        if col in examples.columns
                    ]
                ]
            )

else:
    st.info("Run semantic_search.py to enable topic clustering.")

st.subheader("Signal Distribution")

signal_counts = df["signal"].value_counts().reset_index()
signal_counts.columns = ["signal", "count"]

signal_chart = (
    alt.Chart(signal_counts)
    .mark_bar()
    .encode(
        x=alt.X("signal:N", title="Signal"),
        y=alt.Y("count:Q", title="Count"),
        tooltip=["signal", "count"]
    )
    .properties(height=350)
)

st.altair_chart(signal_chart, use_container_width=True)

if "source" in df.columns:
    st.subheader("News Sources")

    source_counts = df["source"].value_counts().reset_index()
    source_counts.columns = ["source", "count"]

    source_chart = (
        alt.Chart(source_counts)
        .mark_bar()
        .encode(
            x=alt.X("source:N", title="Source"),
            y=alt.Y("count:Q", title="Count"),
            tooltip=["source", "count"]
        )
        .properties(height=350)
    )

    st.altair_chart(source_chart, use_container_width=True)

average_sentiment = round(df["sentiment"].mean(), 2)

if average_sentiment > 0:
    outlook = "Bullish 📈"
elif average_sentiment < 0:
    outlook = "Bearish 📉"
else:
    outlook = "Neutral ➖"

st.subheader("Market Outlook")

col1, col2 = st.columns(2)

with col1:
    st.metric("Overall Market Signal", outlook)

with col2:
    st.metric("Average Sentiment Score", average_sentiment)

if history_path.exists():
    history_df = pd.read_csv(history_path)

    st.subheader("Market Sentiment Over Time")

    history_df["run_time"] = pd.to_datetime(history_df["run_time"])

    sentiment_chart = (
        alt.Chart(history_df)
        .mark_line(color="red", point=True)
        .encode(
            x=alt.X("run_time:T", title="Run Time"),
            y=alt.Y("average_sentiment:Q", title="Average Sentiment"),
            tooltip=["run_time", "average_sentiment"]
        )
        .interactive()
        .properties(height=350)
    )

    st.altair_chart(sentiment_chart, use_container_width=True)

    st.subheader("Historical Run Summary")

    st.dataframe(
        history_df,
        use_container_width=True,
        height=250
    )

st.subheader("Most Bullish Headlines")

bullish = (
    df[df["sentiment"] > 0]
    .sort_values(by="sentiment", ascending=False)
    .head(5)
)

if len(bullish) > 0:
    columns = ["source", "headline", "ticker", "sentiment", "confidence"]
    existing_columns = [col for col in columns if col in bullish.columns]
    st.table(bullish[existing_columns])
else:
    st.info("No bullish headlines found.")

st.subheader("Most Bearish Headlines")

bearish = (
    df[df["sentiment"] < 0]
    .sort_values(by="sentiment")
    .head(5)
)

if len(bearish) > 0:
    columns = ["source", "headline", "ticker", "sentiment", "confidence"]
    existing_columns = [col for col in columns if col in bearish.columns]
    st.table(bearish[existing_columns])
else:
    st.info("No bearish headlines found.")
