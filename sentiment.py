from transformers import pipeline
from utils import detect_coin

classifier = pipeline("sentiment-analysis")

def analyze(news_list):
    results = {}

    for text in news_list:
        coin = detect_coin(text)
        print(text)
        if not coin:
            continue

        output = classifier(text)[0]
        sentiment = output["label"]
        confidence = output["score"]

        if coin not in results:
            results[coin] = []

        results[coin].append({
            "sentiment": sentiment,
            "confidence": confidence
        })

    return results
def aggregate(results):
    final = {}

    for coin, entries in results.items():
        pos = 0
        neg = 0

        for item in entries:
            if item["sentiment"] == "POSITIVE":
                pos += 1
            else:
                neg += 1

        if pos > neg:
            sentiment = "Bullish 📈"
        elif neg > pos:
            sentiment = "Bearish 📉"
        else:
            sentiment = "Neutral 😐"

        avg_confidence = sum(item["confidence"] for item in entries) / len(entries)

        final[coin] = {
            "sentiment": sentiment,
            "confidence": round(avg_confidence, 2)
        }

    return final




