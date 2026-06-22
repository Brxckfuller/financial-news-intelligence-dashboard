from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
import feedparser
import requests


feeds = [
    {
        "source": "MarketWatch",
        "url": "https://feeds.marketwatch.com/marketwatch/topstories/"
    },
    {
        "source": "Investing.com",
        "url": "https://www.investing.com/rss/news.rss"
    },
    {
        "source": "CNBC Markets",
        "url": "https://www.cnbc.com/id/100003114/device/rss/rss.html"
    },
    {
        "source": "CNBC Business",
        "url": "https://www.cnbc.com/id/10001147/device/rss/rss.html"
    },
    {
        "source": "Yahoo Finance",
        "url": "https://finance.yahoo.com/news/rssindex"
    },
    {
        "source": "Nasdaq",
        "url": "https://www.nasdaq.com/feed/rssoutbound"
    }
]


MAX_HEADLINE_AGE_HOURS = 24
REQUEST_TIMEOUT_SECONDS = 10


market_keywords = [
    "stock", "stocks", "market", "markets", "wall street",
    "nasdaq", "s&p", "dow", "shares", "trading", "investors",
    "fed", "federal reserve", "interest rate", "rates",
    "inflation", "recession", "economy", "economic",
    "earnings", "revenue", "profit", "profits", "guidance",
    "bitcoin", "crypto", "oil", "gold", "bond", "bonds",
    "treasury", "yield", "yields", "tariff", "bank", "banks",
    "volatility", "rally", "selloff", "ipo", "merger",
    "acquisition", "dividend", "forecast", "outlook",
    "etf", "fund", "futures", "analyst", "upgrade", "downgrade"
]


blocked_keywords = [
    "inherit", "inheritance", "grandmother", "grandfather",
    "mother", "father", "wife", "husband", "children",
    "college students", "retirement advice", "annuity",
    "estate", "divorce", "social security", "medicare",
    "post office", "mail", "personal finance",
    "credit card", "student loan"
]


def is_market_related(title):
    title_lower = title.lower()

    if any(blocked in title_lower for blocked in blocked_keywords):
        return False

    return any(keyword in title_lower for keyword in market_keywords)


def get_published_datetime(entry):
    if entry.get("published_parsed"):
        return datetime(
            *entry.published_parsed[:6],
            tzinfo=timezone.utc
        )

    if entry.get("updated_parsed"):
        return datetime(
            *entry.updated_parsed[:6],
            tzinfo=timezone.utc
        )

    return None


def is_recent(entry):
    published_time = get_published_datetime(entry)

    if published_time is None:
        return False

    now = datetime.now(timezone.utc)
    age_hours = (now - published_time).total_seconds() / 3600

    return age_hours <= MAX_HEADLINE_AGE_HOURS


def fetch_feed(feed_url):
    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    response = requests.get(
        feed_url,
        headers=headers,
        timeout=REQUEST_TIMEOUT_SECONDS
    )

    response.raise_for_status()

    return feedparser.parse(response.content)


headlines = []

for feed in feeds:
    source = feed["source"]
    feed_url = feed["url"]

    print(f"Reading {source}...")

    try:
        parsed_feed = fetch_feed(feed_url)

        if parsed_feed.bozo:
            print(f"Warning: possible feed issue with {source}")

        for entry in parsed_feed.entries:
            title = entry.get("title", "").strip()

            if not title:
                continue

            if not is_recent(entry):
                continue

            if not is_market_related(title):
                continue

            published_time = get_published_datetime(entry)

            headlines.append(
                {
                    "source": source,
                    "headline": title,
                    "timestamp": published_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "ingested_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
                }
            )

    except requests.exceptions.Timeout:
        print(f"Skipped {source}: request timed out")

    except requests.exceptions.RequestException as error:
        print(f"Skipped {source}: request failed - {error}")

    except Exception as error:
        print(f"Skipped {source}: unexpected error - {error}")


df = pd.DataFrame(headlines)

output_path = Path(__file__).parent.parent / "data" / "news.csv"

if len(df) == 0:
    print("No recent market-related headlines found.")
    print("Existing news.csv was preserved.")
else:
    df = df.drop_duplicates(subset=["headline"])
    df = df.sort_values(by=["timestamp"], ascending=False)

    df.to_csv(output_path, index=False)

    print(f"\nCollected {len(df)} recent market-related headlines")
    print(f"Saved headlines to: {output_path}")
    print(df)