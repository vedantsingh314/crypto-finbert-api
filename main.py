from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sentiment import analyze, aggregate
import time
from news import get_news_for_coin
app = FastAPI()

# ✅ Enable CORS (for Next.js)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Cache: { coin: {data, timestamp} }
cache = {}
CACHE_TTL = 86400  # 1 day


# ✅ Normalize coin names → symbols
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


# ✅ Keywords for filtering
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
    # 🔥 STEP 1 — Normalize input
    coin = coin_map.get(coin.lower(), coin.upper())

    print("REQUESTED COIN:", coin)

    # ✅ STEP 2 — Check cache
    if coin in cache:
        if time.time() - cache[coin]["timestamp"] < CACHE_TTL:
            print("Returning from cache")
            return cache[coin]["data"]

    # ✅ Fetch real news
    news = get_news_for_coin(coin)

    # 🔥 STEP 3 — Filter news correctly
    keywords = coin_keywords.get(coin, [])

    filtered_news = [
        n for n in news
        if any(k in n.lower() for k in keywords)
    ]

    print("FILTERED NEWS COUNT:", len(filtered_news))

    # fallback if nothing found
    if not filtered_news:
        print("No specific news found, using all news")
        filtered_news = news

    # ✅ STEP 4 — Analyze + aggregate
    analyzed = analyze(filtered_news)
    result = aggregate(analyzed)

    print("AGGREGATED RESULT:", result)

    # 🔥 STEP 5 — Extract only requested coin
    coin_result = result.get(coin)

    if not coin_result:
        coin_result = {
            "sentiment": "No data 😐",
            "confidence": 0
        }

    response = {
        coin: coin_result
    }

    print("FINAL RESPONSE:", response)

    # ✅ STEP 6 — Store in cache
    cache[coin] = {
        "data": response,
        "timestamp": time.time()
    }

    return response