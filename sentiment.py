from transformers import pipeline

# ✅ Load model ONCE (very important)
classifier = pipeline("sentiment-analysis")

def detect_coin(text: str):
    text = text.lower()

    if "bitcoin" in text or "btc" in text:
        return "BTC"
    elif "ethereum" in text or "eth" in text:
        return "ETH"
    else:
        return "CRYPTO"


def analyze(news_list):
    if not news_list:
        return {}

    # ✅ Batch processing (fast)
    outputs = classifier(news_list)

    results = {}

    for text, output in zip(news_list, outputs):
        coin = detect_coin(text)

        if coin not in results:
            results[coin] = []

        results[coin].append({
            "sentiment": output["label"],
            "confidence": output["score"]
        })

    return results


def aggregate(results):
    final = {}

    for coin, entries in results.items():
        pos = sum(1 for e in entries if e["sentiment"] == "POSITIVE")
        neg = sum(1 for e in entries if e["sentiment"] == "NEGATIVE")

        if pos > neg:
            sentiment = "Bullish 📈"
        elif neg > pos:
            sentiment = "Bearish 📉"
        else:
            sentiment = "Neutral 😐"

        avg_conf = sum(e["confidence"] for e in entries) / len(entries)

        final[coin] = {
            "sentiment": sentiment,
            "confidence": round(avg_conf, 2)
        }

    return final