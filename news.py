import requests
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("NEWS_API_KEY")
import requests
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta

load_dotenv()

API_KEY = os.getenv("NEWS_API_KEY")


def get_news_for_coin(coin):
    query_map = {
        "BTC": "bitcoin",
        "ETH": "ethereum",
        "SOL": "solana",
        "DOGE": "dogecoin"
    }

    query = query_map.get(coin, "crypto")

    url = f"https://newsapi.org/v2/everything?q={query}&language=en&pageSize=50&apiKey={API_KEY}"

    res = requests.get(url)
    data = res.json()

    articles = []

    for article in data.get("articles", []):
        if article["title"]:
            articles.append(article["title"])

    return articles

