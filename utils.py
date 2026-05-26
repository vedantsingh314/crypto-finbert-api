import re

COIN_MAP = {
    "BTC": [
        "bitcoin", "btc"
    ],

    "ETH": [
        "ethereum", "eth"
    ],

    "BNB": [
        "bnb", "binance coin"
    ],

    "SOL": [
        "solana", "sol"
    ],

    "XRP": [
        "ripple", "xrp"
    ],

    "ADA": [
        "cardano", "ada"
    ],

    "DOGE": [
        "dogecoin", "doge"
    ],

    "AVAX": [
        "avalanche", "avax"
    ],

    "DOT": [
        "polkadot", "dot"
    ],

    "MATIC": [
        "polygon", "matic", "pol"
    ],

    "LINK": [
        "chainlink", "link"
    ],

    "LTC": [
        "litecoin", "ltc"
    ],

    "SHIB": [
        "shiba", "shib", "shiba inu"
    ],

    "TRX": [
        "tron", "trx"
    ],

    "UNI": [
        "uniswap", "uni"
    ],

    "ATOM": [
        "cosmos", "atom"
    ],

    "XLM": [
        "stellar", "xlm"
    ],

    "NEAR": [
        "near", "near protocol"
    ],

    "APT": [
        "aptos", "apt"
    ],

    "ARB": [
        "arbitrum", "arb"
    ],

    "OP": [
        "optimism", "op"
    ],

    "SUI": [
        "sui"
    ],

    "PEPE": [
        "pepe"
    ],

    "INJ": [
        "injective", "inj"
    ],

    "FIL": [
        "filecoin", "fil"
    ],

    "ICP": [
        "internet computer", "icp"
    ],

    "HBAR": [
        "hedera", "hbar"
    ],

    "TIA": [
        "celestia", "tia"
    ],

    "SEI": [
        "sei"
    ],

    "RNDR": [
        "render", "rndr"
    ],

    "FET": [
        "fetch.ai", "fetch", "fet"
    ],

    "TAO": [
        "bittensor", "tao"
    ],

    "CRO": [
        "crypto.com", "cro"
    ],

    "ETC": [
        "ethereum classic", "etc"
    ],

    "BCH": [
        "bitcoin cash", "bch"
    ],

    "XMR": [
        "monero", "xmr"
    ],

    "VET": [
        "vechain", "vet"
    ],

    "AAVE": [
        "aave"
    ],

    "SAND": [
        "sandbox", "sand"
    ],

    "MANA": [
        "decentraland", "mana"
    ],

    "GRT": [
        "the graph", "grt"
    ],

    "ALGO": [
        "algorand", "algo"
    ],

    "EOS": [
        "eos"
    ],

    "FLOW": [
        "flow"
    ],

    "EGLD": [
        "multiversx", "egld", "elrond"
    ],

    "KAS": [
        "kaspa", "kas"
    ],

    "WIF": [
        "dogwifhat", "wif"
    ]
}


def detect_coin(text):
    text = text.lower()

    for symbol, keywords in COIN_MAP.items():
        for keyword in keywords:

            # whole-word regex matching
            pattern = rf"\b{re.escape(keyword.lower())}\b"

            if re.search(pattern, text):
                return symbol

    if re.search(r"\b(crypto|cryptocurrency|digital asset)\b", text):
        return "CRYPTO"

    return None