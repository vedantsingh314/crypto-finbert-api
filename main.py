from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sentiment import analyze, aggregate
from news import get_news_for_coin
import time

app = FastAPI()

# ✅ Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Cache
cache = {}
CACHE_TTL = 86400  # 1 day


# ✅ Normalize coin names
coin_map = {
    "bitcoin": "BTC",
    "btc": "BTC",
    "ethereum": "ETH",
    "eth": "ETH",
    "solana": "SOL",
    "sol": "SOL",
    "dogecoin": "DOGE",
    "doge": "DOGE"
}


# ✅ Keywords
coin_keywords = {
    "BTC": ["bitcoin", "btc"],
    "ETH": ["ethereum", "eth"],
    "SOL": ["solana", "sol"],
    "DOGE": ["dogecoin", "doge"]
}


@app.get("/")
def home():
    return {"message": "Crypto Sentiment API running 🚀"}


@app.get("/coin-sentiment")
def coin_sentiment(coin: str):
    try:
        # 🔥 STEP 1 — Normalize
        coin = coin_map.get(coin.lower(), coin.upper())
        print("REQUESTED COIN:", coin)

        # ✅ STEP 2 — Cache
        if coin in cache:
            if time.time() - cache[coin]["timestamp"] < CACHE_TTL:
                print("Returning from cache")
                return cache[coin]["data"]

        # ✅ STEP 3 — Fetch news
        news = get_news_for_coin(coin)

        if not news:
            return {coin: {"sentiment": "No data 😐", "confidence": 0}}

        # 🔥 STEP 4 — Filter news
        keywords = coin_keywords.get(coin, [])

        filtered_news = [
            n for n in news
            if any(k in n.lower() for k in keywords)
        ]

        print("FILTERED NEWS COUNT:", len(filtered_news))

        if not filtered_news:
            print("No specific news found, using all news")
            filtered_news = news

        # 🚀 LIMIT (VERY IMPORTANT for speed)
        filtered_news = filtered_news[:10]

        # ✅ STEP 5 — Analyze
        analyzed = analyze(filtered_news)

        if not analyzed:
            return {coin: {"sentiment": "No data 😐", "confidence": 0}}

        result = aggregate(analyzed)

        print("AGGREGATED RESULT:", result)

        # 🔥 STEP 6 — Extract coin
        coin_result = result.get(coin, {
            "sentiment": "No data 😐",
            "confidence": 0
        })

        response = {coin: coin_result}

        print("FINAL RESPONSE:", response)

        # ✅ STEP 7 — Cache
        cache[coin] = {
            "data": response,
            "timestamp": time.time()
        }

        return response

    except Exception as e:
        print("ERROR:", e)
        return {"error": str(e)}