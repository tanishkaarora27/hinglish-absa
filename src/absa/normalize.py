"""
Stage 1 - Text normalisation for Romanised Hinglish.

Why this matters (viva answer):
    Hinglish has no fixed spelling. The same word appears as "accha", "acha",
    "achha", "achaa", "acchaaa". Social-media text also has elongations
    ("soooo good"), slang ("bkwas", "mst"), emoji and mixed case. If we feed raw
    text to a model, every spelling variant becomes a *different* feature and
    the model learns nothing general. Normalisation collapses these variants so
    the same meaning maps to the same token.

What we do, in order:
    1. Unicode normalise + lowercase.
    2. Replace URLs / @mentions / numbers with placeholder tokens.
    3. Translate emoji into words (":thumbs_up:" -> "emoji_thumbs_up") so emoji
       become usable sentiment features instead of being thrown away.
    4. Collapse elongations: any character repeated 3+ times -> 2 copies
       ("soooo" -> "soo"), and record how many elongations we saw (this count is
       later a sarcasm feature - people elongate when exaggerating).
    5. Map known spelling variants / slang to a canonical form using a small
       hand-built dictionary (e.g. "acha" -> "accha", "bkwas" -> "bakwas").
    6. Tokenise on whitespace and punctuation; keep "!" and "?" as tokens
       because they carry emotion.

Everything here is rule-based and transparent, so it can be explained and
extended by hand.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import List

try:
    import emoji as _emoji
except ImportError:  # keep the module importable even without the emoji package
    _emoji = None

# ---------------------------------------------------------------------------
# 5. Spelling-variant / slang dictionary.
# Keys are observed variants, values are the canonical form we want to keep.
# This list was written by hand from common Hinglish spellings. Extend freely
# after looking at your own data (notebook shows the most frequent unknown
# tokens so you can add them).
# ---------------------------------------------------------------------------
SPELLING_MAP = {
    # good / bad
    "acha": "accha", "achha": "accha", "achaa": "accha", "aca": "accha",
    "acche": "accha", "ache": "accha", "achhe": "accha", "acchi": "accha",
    "achi": "accha", "achhi": "accha", "badiya": "badhiya", "badhia": "badhiya",
    "bdhiya": "badhiya", "mst": "mast", "masst": "mast", "zabardast": "zabardast",
    "jabardast": "zabardast", "jhakaas": "jhakaas", "jhakas": "jhakaas",
    "bkwas": "bakwas", "bakwaas": "bakwas", "bakvas": "bakwas", "bekar": "bekaar",
    "bekr": "bekaar", "ghatiya": "ghatiya", "ghatia": "ghatiya", "kharab": "kharab",
    "khrab": "kharab", "kharaab": "kharab", "faltu": "faltu", "fltu": "faltu",
    "waahiyat": "wahiyat", "wahiyaat": "wahiyat",
    # time / speed
    "jldi": "jaldi", "jaldee": "jaldi", "jaldii": "jaldi", "der": "der",
    "dair": "der", "deri": "der", "late": "late", "lt": "late",
    "minit": "minute", "mint": "minute", "mints": "minute", "mins": "minute", "min": "minute", "minutes": "minute",
    "ghanta": "ghanta", "ghante": "ghanta", "ghnta": "ghanta", "hrs": "hour",
    "hr": "hour", "hours": "hour",
    # negation / function words
    "nhi": "nahi", "nahin": "nahi", "nai": "nahi", "nhii": "nahi", "nahii": "nahi",
    "nhin": "nahi", "ni": "nahi", "mat": "mat", "kbhi": "kabhi", "kabhee": "kabhi",
    "bilkul": "bilkul", "bilkuul": "bilkul", "bilkl": "bilkul",
    "bhut": "bahut", "bohot": "bahut", "bhot": "bahut", "bahot": "bahut",
    "bohut": "bahut", "bht": "bahut", "boht": "bahut", "bahuut": "bahut",
    "h": "hai", "hain": "hai", "hy": "hai", "hae": "hai", "he": "hai",
    "kya": "kya", "kyaa": "kya", "kiya": "kiya", "q": "kyu", "kyun": "kyu",
    "kyon": "kyu", "kyo": "kyu", "ku": "kyu",
    "plz": "please", "pls": "please", "plzz": "please",
    "thnx": "thanks", "thx": "thanks", "thnks": "thanks", "thank": "thanks",
    "u": "you", "ur": "your", "r": "are", "y": "why", "n": "and", "nd": "and",
    "wid": "with", "wth": "with", "dnt": "dont", "don't": "dont", "didn't": "didnt",
    "doesn't": "doesnt", "isn't": "isnt", "wasn't": "wasnt", "can't": "cant",
    "won't": "wont", "haven't": "havent", "hasn't": "hasnt", "couldn't": "couldnt",
    "wouldn't": "wouldnt", "never": "never", "nvr": "never",
    # domain nouns
    "delivry": "delivery", "dilivery": "delivery", "delevery": "delivery",
    "delivary": "delivery", "deliver": "delivery", "delivered": "delivery",
    "delivering": "delivery", "deliveries": "delivery",
    "pkging": "packaging", "packing": "packaging", "packed": "packaging",
    "pack": "packaging", "packet": "packaging",
    "qty": "quantity", "qlty": "quality", "qualty": "quality", "kwality": "quality",
    "mehnga": "mehenga", "mehanga": "mehenga", "mahenga": "mehenga",
    "mehngi": "mehenga", "mehengi": "mehenga", "mahngi": "mehenga",
    "sasta": "sasta", "sasti": "sasta", "saste": "sasta",
    "paisa": "paise", "paisay": "paise", "rs": "rupees", "rs.": "rupees",
    "rupay": "rupees", "rupaye": "rupees", "rupee": "rupees",
    "sabji": "sabzi", "sabjee": "sabzi", "sabzee": "sabzi", "sabziyan": "sabzi",
    "sabjiyan": "sabzi", "veggies": "vegetables", "veg": "vegetables",
    "fruit": "fruits", "frut": "fruits", "froot": "fruits",
    "dudh": "doodh", "milk": "milk", "bread": "bread",
    "taaza": "taza", "taja": "taza", "tazi": "taza", "tazaa": "taza",
    "sada": "sada", "sadi": "sada", "sade": "sada", "sadha": "sada",
    "expiry": "expired", "expire": "expired", "expierd": "expired",
    "toota": "tuta", "tuti": "tuta", "tootaa": "tuta", "tute": "tuta",
    "phata": "phata", "phati": "phata", "fata": "phata", "fati": "phata",
    "bhaiya": "bhaiya", "bhaiyya": "bhaiya", "bhaiyaa": "bhaiya", "bhai": "bhaiya",
    "bhaai": "bhaiya", "bhaia": "bhaiya",
    "customer": "customer", "custmer": "customer", "costumer": "customer",
    "suport": "support", "sport": "support", "supprt": "support",
    "refnd": "refund", "refunded": "refund", "refunds": "refund",
    "aap": "aap", "ap": "aap", "app": "app", "apps": "app", "aplication": "app",
    "application": "app",
    "ordr": "order", "oder": "order", "ordered": "order", "orders": "order",
    "cancle": "cancel", "cancelled": "cancel", "canceled": "cancel",
    "cancellation": "cancel", "cncl": "cancel",
    "awsm": "awesome", "awsome": "awesome", "awesum": "awesome",
    "gud": "good", "gd": "good", "goood": "good", "gr8": "great", "grt": "great",
    "wrst": "worst", "wrost": "worst", "baddd": "bad", "bd": "bad",
    "luv": "love", "lv": "love", "lov": "love",
    "superb": "superb", "supr": "super", "suprb": "superb",
    "wow": "wow", "waah": "wah", "waow": "wow", "vah": "wah", "wahh": "wah",
    "lol": "lol", "lmao": "lol", "rofl": "lol", "hehe": "haha", "hehehe": "haha",
    "hahaha": "haha", "hahah": "haha",
    "pathetic": "pathetic", "pathatic": "pathetic", "horrible": "horrible",
    "horible": "horrible", "terrible": "terrible", "terible": "terrible",
    "disgusting": "disgusting", "disgustng": "disgusting",
    "amazng": "amazing", "amzing": "amazing",
    "timly": "timely", "ontime": "on_time", "on-time": "on_time",
    "ntwrk": "network", "netwrk": "network",
    "behavior": "behaviour", "behaviur": "behaviour", "bhaviour": "behaviour",
    "cud": "could", "shud": "should", "wud": "would",
    "sab": "sab", "sb": "sab", "sabse": "sabse", "sbse": "sabse",
    "jyada": "zyada", "jada": "zyada", "zada": "zyada", "ziada": "zyada",
    "kam": "kam", "km": "kam", "kum": "kam",
    "thoda": "thoda", "thora": "thoda", "thodi": "thoda",
    "ek": "ek", "1": "ek_num",
    "hamesha": "hamesha", "hmesha": "hamesha", "humesha": "hamesha",
    "wapas": "wapas", "vapas": "wapas", "wapis": "wapas", "vapis": "wapas",
    "galat": "galat", "glt": "galat", "ghalat": "galat",
    "sahi": "sahi", "shi": "sahi", "sahii": "sahi", "sahee": "sahi",
    "theek": "theek", "thik": "theek", "thk": "theek", "teek": "theek",
    "paisa_vasool": "paisa_vasool", "paisavasool": "paisa_vasool",
}

# Multi-word phrases we want to keep together as one token ("paisa vasool" is one
# concept). Applied before tokenisation.
PHRASE_MAP = {
    "paisa vasool": "paisa_vasool", "paise vasool": "paisa_vasool", "on time": "on_time",
    "delivery boy": "delivery_partner", "delivery partner": "delivery_partner",
    "delivery guy": "delivery_partner", "delivery person": "delivery_partner",
    "delivery agent": "delivery_partner", "delivery wala": "delivery_partner",
    "delivery wale": "delivery_partner", "delivery executive": "delivery_partner",
    "rider": "delivery_partner", "customer care": "customer_support",
    "customer support": "customer_support", "customer service": "customer_support",
    "help center": "customer_support", "help centre": "customer_support",
    "support team": "customer_support", "chat support": "customer_support",
    "kya baat": "kya_baat", "kya baat hai": "kya_baat", "wah bhai": "wah_bhai",
    "waah bhai": "wah_bhai", "10 minute": "ten_minute", "10 min": "ten_minute",
    "10 mins": "ten_minute", "ten minute": "ten_minute", "ten mins": "ten_minute",
    "handling charge": "handling_fee", "handling charges": "handling_fee",
    "handling fee": "handling_fee", "surge fee": "surge_fee", "surge charge": "surge_fee",
    "delivery charge": "delivery_fee", "delivery charges": "delivery_fee",
    "delivery fee": "delivery_fee", "rain fee": "surge_fee",
    "small cart fee": "small_cart_fee", "delivery partners": "delivery_partner",
    "delivery agents": "delivery_partner", "delivery boys": "delivery_partner",
    "delivery guys": "delivery_partner", "delivery persons": "delivery_partner",
    "delivery man": "delivery_partner", "delivery men": "delivery_partner",
    "delivery executives": "delivery_partner", "delivery staff": "delivery_partner",
    "riders": "delivery_partner", "carry bag": "carry_bag", "paper bag": "carry_bag",
    "paper bags": "carry_bag", "user friendly": "user_friendly",
    "user-friendly": "user_friendly", "help desk": "customer_support", "chat bot": "chatbot",
    "ai bot": "chatbot", "ai chatbot": "chatbot", "ai agent": "chatbot",
    "ai assistant": "chatbot", "cash on delivery": "cod", "free delivery": "delivery_fee",
    "minimum order": "min_order", "platform fee": "handling_fee",
    "platform fees": "handling_fee", "cancellation fee": "cancel_fee",
    "cancellation charges": "cancel_fee", "cancellation charge": "cancel_fee",
    "cancel fee": "cancel_fee", "delivery time": "delivery_time",
    "delivery timing": "delivery_time", "delivery timings": "delivery_time",
    "in time": "on_time", "late delivery": "late_delivery",
    "delayed delivery": "late_delivery", "fast delivery": "fast_delivery",
    "quick delivery": "fast_delivery", "instant delivery": "fast_delivery",
    "superfast delivery": "fast_delivery", "super fast delivery": "fast_delivery",
    "express delivery": "fast_delivery", "fastest delivery": "fast_delivery",
    "speed delivery": "fast_delivery", "delivery speed": "delivery_time",
    "delivery is fast": "fast_delivery", "delivery is slow": "late_delivery",
    "delivery was fast": "fast_delivery", "delivery late": "late_delivery",
    "delivery delayed": "late_delivery", "delivery slow": "late_delivery",
    "delivery fast": "fast_delivery", "dilevery": "delivery", "delevery": "delivery",
    "dilivery": "delivery", "wrong item": "wrong_item", "wrong items": "wrong_item",
    "wrong product": "wrong_item", "wrong order": "wrong_item", "missing item": "missing_item",
    "missing items": "missing_item", "item missing": "missing_item",
    "items missing": "missing_item", "not delivered": "not_delivered",
    "nahi aaya": "not_delivered", "nhi aaya": "not_delivered", "nahi mila": "not_delivered",
    "nhi mila": "not_delivered", "hidden charges": "handling_fee",
    "extra charges": "handling_fee", "extra charge": "handling_fee",
    "handling fees": "handling_fee", "surge fees": "surge_fee", "surge charges": "surge_fee",
    "rain fees": "surge_fee", "delivery fees": "delivery_fee", "ice cream": "ice_cream",
    "icecream": "ice_cream", "ice creams": "ice_cream", "cold drink": "cold_drink",
    "cold drinks": "cold_drink", "near expiry": "expired", "about to expire": "expired",
    "close to expiry": "expired", "expiry date": "expiry_date", "out of stock": "out_of_stock",
    "not available": "not_available", "nahi milta": "not_available",
    "food quality": "food_quality", "quality of food": "food_quality",
    "food was cold": "cold_food", "cold food": "cold_food", "food is cold": "cold_food",
    "stale food": "stale_food", "spoiled food": "stale_food", "rotten food": "stale_food",
    "fresh food": "fresh_food", "food fresh": "fresh_food", "food hot": "hot_food",
    "hot food": "hot_food", "good food": "good_food", "food good": "good_food",
    "food is good": "good_food", "tasty food": "good_food", "bad food": "bad_food",
    "food bad": "bad_food", "food is bad": "bad_food", "less quantity": "less_quantity",
    "quantity less": "less_quantity", "quantity kam": "less_quantity",
    "kam quantity": "less_quantity", "low quantity": "less_quantity", "open box": "open_box",
    "opened pack": "open_box", "opened packet": "open_box", "half open": "open_box",
    "seal broken": "open_box", "seal tuta": "open_box", "box damaged": "damaged_box",
    "broken box": "damaged_box", "damaged packet": "damaged_box", "packet torn": "damaged_box",
    "torn packet": "damaged_box", "bag torn": "damaged_box", "torn bag": "damaged_box",
    "extra money": "handling_fee", "extra paise": "handling_fee", "come down": "come_down",
    "come downstairs": "come_down", "neeche aao": "come_down", "niche aao": "come_down",
    "niche aakar": "come_down",
}

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
NUM_RE = re.compile(r"(?<![a-z_])\d+(?:[.,]\d+)?(?![a-z_])")
ELONG_RE = re.compile(r"(.)\1{2,}")          # any char repeated 3+ times
PUNCT_KEEP_RE = re.compile(r"[!?]+")        # emotional punctuation we keep
TOKEN_RE = re.compile(r"[a-z_]+(?:'[a-z]+)?|<num>|<url>|<user>|!|\?")


@dataclass
class Normalized:
    """Result of normalising one review."""
    raw: str
    text: str                      # normalised string (space separated tokens)
    tokens: List[str] = field(default_factory=list)
    n_elongations: int = 0         # how many elongated words were found
    n_exclaim: int = 0             # number of '!' characters in the raw text
    n_question: int = 0
    n_caps_words: int = 0          # ALL-CAPS words in raw (shouting)
    emojis: List[str] = field(default_factory=list)   # emoji names found


def demojize(text: str) -> tuple[str, List[str]]:
    """Convert emoji to word tokens like 'emoji_thumbs_up'. Returns (text, names)."""
    if _emoji is None:
        return text, []
    names: List[str] = []

    def _repl(match):
        name = _emoji.demojize(match.group(0)).strip(":")
        name = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
        names.append(name)
        return f" emoji_{name} "

    out = _emoji.replace_emoji(text, replace=lambda ch, data: _repl(re.match(r".+", ch, re.S)))
    return out, names


def normalize(text: str) -> Normalized:
    """Full normalisation of a single review. See module docstring for steps."""
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    raw = text
    n_exclaim = raw.count("!")
    n_question = raw.count("?")
    n_caps_words = len([w for w in re.findall(r"\b[A-Z]{3,}\b", raw)])

    t = unicodedata.normalize("NFKC", raw)
    t, emoji_names = demojize(t)
    t = t.lower()
    t = URL_RE.sub(" <url> ", t)
    t = MENTION_RE.sub(" <user> ", t)

    # elongation: count then collapse ("soooo" -> "soo"). We keep TWO copies so
    # that "good" and "goood" still differ by one char, which char n-grams can use
    # as a mild emphasis signal, while "gooooood" and "goood" become identical.
    n_elong = len(ELONG_RE.findall(t))
    t = ELONG_RE.sub(r"\1\1", t)

    # keep emotional punctuation as separate tokens, drop the rest
    t = PUNCT_KEEP_RE.sub(lambda m: f" {m.group(0)[0]} ", t)
    t = re.sub(r"[^a-z0-9_<>!?'\s]", " ", t)

    # phrase map (longest phrases first so "kya baat hai" wins over "kya baat").
    # Runs BEFORE number replacement so "10 min" can become "ten_minute".
    for phrase in sorted(PHRASE_MAP, key=len, reverse=True):
        t = re.sub(rf"\b{re.escape(phrase)}\b", PHRASE_MAP[phrase], t)
    t = NUM_RE.sub(" <num> ", t)

    # token-level spelling map
    toks = []
    for tok in TOKEN_RE.findall(t):
        tok = SPELLING_MAP.get(tok, tok)
        # second pass: after collapsing elongation a variant may now be in the map
        tok = SPELLING_MAP.get(tok, tok)
        toks.append(tok)

    # second phrase pass: spelling fixes may have created a phrase ("delivry boy" -> "delivery boy")
    joined = " ".join(toks)
    for phrase in sorted(PHRASE_MAP, key=len, reverse=True):
        joined = re.sub(rf"\b{re.escape(phrase)}\b", PHRASE_MAP[phrase], joined)
    toks = joined.split()

    return Normalized(raw=raw, text=" ".join(toks), tokens=toks,
                      n_elongations=n_elong, n_exclaim=n_exclaim,
                      n_question=n_question, n_caps_words=n_caps_words,
                      emojis=emoji_names)


def normalize_text(text: str) -> str:
    """Convenience: return only the normalised string."""
    return normalize(text).text


if __name__ == "__main__":  # quick self-check
    for s in ["Delivery sooooo late thi, 10 min bola 1 ghanta laga 😡 bkwas service!!",
              "Wah bhai kya baat hai, doodh expired bhej diya 👏 great job Blinkit"]:
        n = normalize(s)
        print(n.text, "| elong:", n.n_elongations, "| !:", n.n_exclaim, "| emoji:", n.emojis)
