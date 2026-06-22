from pathlib import Path
from datetime import datetime
import re
import pandas as pd
from transformers import pipeline

base_path = Path(__file__).parent.parent

input_path = base_path / "data" / "news.csv"
output_path = base_path / "data" / "analysed_news.csv"
history_path = base_path / "data" / "sentiment_history.csv"

df = pd.read_csv(input_path)

known_tickers = {
    "apple": "AAPL",
    "microsoft": "MSFT",
    "nvidia": "NVDA",
    "tesla": "TSLA",
    "amazon": "AMZN",
    "google": "GOOGL",
    "alphabet": "GOOGL",
    "meta": "META",
    "netflix": "NFLX",
    "bitcoin": "BTC",
    "ethereum": "ETH",
    "carmax": "KMX",
    "gitlab": "GTLB",
    "ebay": "EBAY",
    "quest diagnostics": "DGX",
    "tower semiconductor": "TSEM",
    "relmada therapeutics": "RLMD",
    "compass": "COMP",
    "cytomx": "CTMX",
    "citi": "C",
    "abbott": "ABT",
    "amd": "AMD",
    "nike": "NKE",
}


def extract_ticker(headline):
    headline_text = str(headline)
    headline_lower = headline_text.lower()

    matches = re.findall(r"\(([A-Z]{1,5})\)", headline_text)

    if matches:
        return matches[0]

    for company_name, ticker in known_tickers.items():
        if company_name in headline_lower:
            return ticker

    return "UNKNOWN"


print("Loading FinBERT model...")

classifier = pipeline(
    "sentiment-analysis",
    model="ProsusAI/finbert"
)

print("Analysing headlines...")


def analyse_sentiment(headline):
    result = classifier(str(headline))[0]

    label = result["label"].lower()
    confidence = result["score"]

    if label == "positive":
        sentiment = round(confidence, 3)
        signal = "Bullish"
    elif label == "negative":
        sentiment = round(-confidence, 3)
        signal = "Bearish"
    else:
        sentiment = 0
        signal = "Neutral"

    return pd.Series({
        "sentiment": sentiment,
        "signal": signal,
        "confidence": round(confidence, 3)
    })


df["ticker"] = df["headline"].apply(extract_ticker)

sentiment_results = df["headline"].apply(analyse_sentiment)

df = pd.concat([df, sentiment_results], axis=1)

df = df.sort_values(by="sentiment", ascending=False)

df.to_csv(output_path, index=False)

# Save historical run summary


run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

history_row = pd.DataFrame([{
    "run_time": run_time,
    "headline_count": len(df),
    "bullish_count": len(df[df["signal"] == "Bullish"]),
    "bearish_count": len(df[df["signal"] == "Bearish"]),
    "neutral_count": len(df[df["signal"] == "Neutral"]),
    "average_sentiment": round(df["sentiment"].mean(), 3)
}])

if history_path.exists():
    history_df = pd.read_csv(history_path)
    history_df = pd.concat([history_df, history_row], ignore_index=True)
else:
    history_df = history_row

history_df.to_csv(history_path, index=False)

print(df)
print(f"\nSaved analysed data to: {output_path}")
print(f"Saved sentiment history to: {history_path}")