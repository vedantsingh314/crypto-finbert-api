def detect_coin(text):
    text = text.lower()

    if "bitcoin" in text or "btc" in text:
        return "BTC"
    if "ethereum" in text or "eth" in text:
        return "ETH"
    if "crypto" in text:
        return "CRYPTO"   # fallback category

    return None