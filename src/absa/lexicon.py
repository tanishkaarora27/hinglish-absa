"""
Seed lexicons (hand-built, transparent, extendable).

Why lexicons at all (viva answer):
    We have little labelled data. A small list of domain words lets us
    (1) find WHICH aspect a review talks about (aspect seed lexicon), and
    (2) compute simple sentiment counts that feed the sarcasm detector
    (sentiment lexicon). Lexicon rules are cheap, explainable, and work from
    day one; the ML models then learn what the rules miss.

All words are in the NORMALISED form produced by normalize.py (so write
"accha", not "acha"; "mehenga", not "mehnga").
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Aspect seed lexicon. Keys = aspect category. Values = trigger tokens / phrases
# (already normalised). Multi-token phrases are allowed ("late delivery").
# ---------------------------------------------------------------------------
ASPECT_LEXICON = {
    "delivery_speed": [
        "late", "der", "jaldi", "fast", "slow", "on_time", "ten_minute", "minute", "ghanta",
        "hour", "wait", "waiting", "delay", "delayed", "quick", "quickly", "speed", "turant",
        "pahuncha", "pahunch", "aaya", "aayi", "ayi", "eta", "timely", "superfast", "instant",
        "deri", "intezaar", "intezar", "delivery_time", "late_delivery", "fast_delivery",
        "hours", "hrs", "waited", "faster", "fastest", "early", "promptly", "not_delivered",
        "arrive", "arriving", "arrival",
    ],
    "product_quality": [
        "quality", "fresh", "taza", "sada", "rotten", "expired", "kharab", "stale", "basi",
        "smell", "badbu", "ghatiya", "fruits", "sabzi", "vegetables", "doodh", "milk", "bread",
        "egg", "eggs", "anda", "ande", "paneer", "curd", "dahi", "chicken", "meat", "fish",
        "saman", "samaan", "cheez", "damaged", "spoiled", "spoilt", "fake", "duplicate",
        "brand", "taste", "swad", "kadwa", "khatta", "sadke", "galat", "missing", "ripe",
        "unripe", "kacha", "kachha", "pakka", "crushed", "mashed", "melted", "pighal",
        "quantity", "condition", "hot", "warm", "chilled", "ice_cream", "cold_drink",
        "wrong_item", "missing_item", "expiry_date", "tasty", "dirty", "dusty", "hygiene",
        "hygienic", "unhygienic", "insects", "insect", "worms", "fungus", "kida", "keede",
        "ganda", "gandi", "sadi", "sade", "freshness", "defective", "genuine", "dented",
        "curdled", "smelling", "chhota", "food_quality", "cold_food", "stale_food",
        "fresh_food", "hot_food", "good_food", "bad_food", "less_quantity",
    ],
    "packaging": [
        "packaging", "packet", "pack", "seal", "sealed", "leak", "leaked", "leaking", "spill",
        "spilled", "tuta", "phata", "torn", "bag", "wrapped", "wrapping", "cover", "dabba",
        "carry_bag", "polythene", "plastic", "polybag", "paper", "bags", "insulated", "packed",
        "packing", "package", "unpacked", "open_box", "damaged_box",
    ],
    "price": [
        "price", "prices", "mehenga", "expensive", "costly", "sasta", "cheap", "charges",
        "charge", "fee", "fees", "handling_fee", "surge_fee", "delivery_fee", "small_cart_fee",
        "rupees", "mrp", "discount", "offer", "offers", "coupon", "cashback", "rate", "rates",
        "paisa_vasool", "loot", "lootte", "overpriced", "value", "cost", "hidden", "bill",
        "minimum", "min_order", "discounts", "coupons", "charged", "charging", "cheaper",
        "cheapest", "affordable", "reasonable", "pricing", "deal", "deals", "save", "savings",
        "saving", "gst", "tax", "cancel_fee",
    ],
    "delivery_partner": [
        "delivery_partner", "uncle", "boy", "rude", "polite", "behaviour", "misbehave",
        "misbehaved", "badtameez", "badtamiz", "tameez", "shouted", "chilla", "chillaya",
        "ring", "doorbell", "gate", "bhaga", "nice guy", "sweet", "partners", "partner",
        "rider", "riders", "boys", "guy", "behavior", "behaved", "rudely", "politely",
        "courteous", "misbehaving", "argued", "arguing", "threat", "threatened", "abuse",
        "abused", "downstairs", "upstairs", "stairs", "floor", "come down", "chutta",
        "chillar", "drunk", "reckless", "helmet", "come_down",
    ],
    "app_support": [
        "app", "crash", "crashed", "crashing", "hang", "hangs", "lag", "laggy", "bug", "bugs",
        "glitch", "login", "otp", "payment", "upi", "wallet", "refund", "customer_support",
        "support", "chat", "chatbot", "bot", "executive", "complaint", "complain", "ticket",
        "response", "respond", "reply", "resolution", "resolve", "resolved", "cancel",
        "cancelled", "tracking", "track", "map", "notification", "ui", "interface", "search",
        "cart", "checkout", "coupon code", "error", "server", "loading", "load", "network",
        "no reply", "option", "options", "user_friendly", "ux", "navigate", "navigation",
        "feature", "features", "buggy", "glitches", "crashes", "account", "payments",
        "refunds", "refunded", "replacement", "replace", "exchange", "canceled",
        "cancellation", "helpline", "agent", "agents", "executives", "email", "mail", "mailed",
        "contact", "connect", "responded", "replied", "resolving", "complaints", "query",
        "queries", "issue", "issues", "notifications", "ads", "popup", "popups", "pop",
        "application", "apps", "install", "installed", "uninstall", "uninstalled",
        "uninstalling", "reinstall", "errors", "service unavailable", "unserviceable",
        "not_available", "out_of_stock", "availability", "unavailable", "serviceable",
        "coverage",
    ],
}
# NOTE: generic words ('delivery', 'time', 'order', 'item', 'food', 'money') were REMOVED as
# stand-alone triggers after measuring precision on the labelled data; they are now only
# matched inside disambiguating phrases (see normalize.PHRASE_MAP), e.g. 'late delivery',
# 'food quality', 'extra money'. Lexicon v2: micro-F1 vs gold aspects 0.687 -> 0.758.

# Clause boundaries: a context window never crosses these (viva: "lekin / but"
# flips sentiment, so the window must stop there).
CLAUSE_BREAKERS = {"but", "lekin", "par", "magar", "kintu", "however", "although",
                   "though", "while", "whereas", "jabki", "phir", "fir", "then",
                   "baaki", "baki", "otherwise", "warna", "varna", "!", "?"}

# ---------------------------------------------------------------------------
# Sentiment lexicon (normalised tokens). Scores in [-1, 1]. Used for
# (a) incongruity features in the sarcasm detector,
# (b) a fallback / sanity check for sentiment.
# ---------------------------------------------------------------------------
POSITIVE = {
    "accha": 0.8, "badhiya": 0.9, "mast": 0.9, "zabardast": 1.0, "jhakaas": 1.0,
    "good": 0.7, "great": 0.9, "awesome": 1.0, "amazing": 1.0, "excellent": 1.0,
    "superb": 1.0, "super": 0.8, "best": 0.9, "nice": 0.7, "love": 0.9, "loved": 0.9,
    "fresh": 0.8, "taza": 0.8, "fast": 0.7, "quick": 0.7, "quickly": 0.7, "jaldi": 0.7,
    "on_time": 0.8, "timely": 0.8, "ten_minute": 0.5, "superfast": 0.9, "instant": 0.7,
    "sasta": 0.7, "cheap": 0.5, "paisa_vasool": 1.0, "worth": 0.7, "value": 0.5,
    "polite": 0.9, "helpful": 0.9, "humble": 0.8, "friendly": 0.8, "sweet": 0.6,
    "happy": 0.8, "satisfied": 0.9, "satisfying": 0.8, "perfect": 1.0, "smooth": 0.8,
    "easy": 0.6, "convenient": 0.8, "recommend": 0.8, "recommended": 0.8,
    "thanks": 0.6, "thank": 0.6, "wow": 0.7, "wah": 0.7, "kya_baat": 0.8,
    "wah_bhai": 0.8, "sahi": 0.6, "theek": 0.4, "fine": 0.4, "ok": 0.3, "okay": 0.3,
    "sealed": 0.4, "proper": 0.6, "properly": 0.6, "clean": 0.6, "hygienic": 0.7,
    "reliable": 0.8, "resolved": 0.7, "refund": 0.2, "discount": 0.5, "offer": 0.4,
    "cashback": 0.5, "lifesaver": 1.0, "fantastic": 1.0, "brilliant": 1.0,
    "shandar": 1.0, "shaandar": 1.0, "kamaal": 0.9, "kamal": 0.9, "bawal": 0.9,
    "bawaal": 0.9, "op": 0.8, "lit": 0.7, "emoji_thumbs_up": 0.7, "emoji_red_heart": 0.8,
    "emoji_smiling_face_with_smiling_eyes": 0.7, "emoji_grinning_face": 0.6,
    "emoji_star_struck": 0.9, "emoji_fire": 0.7, "emoji_clapping_hands": 0.6,
    "emoji_folded_hands": 0.4, "emoji_ok_hand": 0.6, "emoji_smiling_face_with_heart_eyes": 0.9,
    "emoji_party_popper": 0.8, "emoji_hundred_points": 0.8, "emoji_sparkles": 0.6,
}
NEGATIVE = {
    "bakwas": -1.0, "bekaar": -0.9, "ghatiya": -1.0, "kharab": -0.8, "faltu": -0.8,
    "wahiyat": -1.0, "bad": -0.7, "worst": -1.0, "poor": -0.7, "pathetic": -1.0,
    "horrible": -1.0, "terrible": -1.0, "disgusting": -1.0, "useless": -0.9,
    "waste": -0.8, "fraud": -1.0, "scam": -1.0, "cheat": -0.9, "cheated": -0.9,
    "loot": -0.9, "lootte": -0.9, "late": -0.7, "der": -0.6, "slow": -0.6,
    "delay": -0.7, "delayed": -0.7, "wait": -0.3, "waiting": -0.4, "intezaar": -0.4,
    "expired": -1.0, "rotten": -1.0, "sada": -0.9, "stale": -0.8, "basi": -0.8,
    "smell": -0.5, "badbu": -0.8, "spoiled": -0.9, "spoilt": -0.9, "damaged": -0.8,
    "broken": -0.8, "tuta": -0.8, "phata": -0.7, "leak": -0.7, "leaked": -0.7,
    "leaking": -0.7, "spill": -0.6, "spilled": -0.6, "missing": -0.8, "wrong": -0.7,
    "galat": -0.7, "fake": -1.0, "duplicate": -0.8, "mehenga": -0.7, "expensive": -0.6,
    "costly": -0.6, "overpriced": -0.8, "hidden": -0.5, "extra": -0.3, "rude": -0.9,
    "misbehave": -1.0, "misbehaved": -1.0, "badtameez": -1.0, "badtamiz": -1.0,
    "shouted": -0.8, "chilla": -0.7, "chillaya": -0.7, "attitude": -0.5,
    "crash": -0.8, "crashed": -0.8, "crashing": -0.8, "hang": -0.6, "hangs": -0.6,
    "lag": -0.5, "laggy": -0.6, "bug": -0.6, "bugs": -0.6, "glitch": -0.6, "error": -0.5,
    "cancel": -0.5, "cancelled": -0.6, "complaint": -0.4, "no": -0.3, "never": -0.6,
    "nahi": -0.2, "angry": -0.8, "frustrated": -0.8, "frustrating": -0.8,
    "disappointed": -0.8, "disappointing": -0.8, "unhappy": -0.8, "sad": -0.6,
    "pareshan": -0.7, "dukhi": -0.7, "gussa": -0.7, "hopeless": -0.9, "nonsense": -0.8,
    "ridiculous": -0.8, "uninstall": -0.9, "uninstalled": -0.9, "uninstalling": -0.9,
    "deleted": -0.6, "delete": -0.5, "switching": -0.5, "switched": -0.5, "zepto": 0.0,
    "regret": -0.8, "unprofessional": -0.8, "careless": -0.7, "ignore": -0.6,
    "ignored": -0.7, "ignoring": -0.7, "harassment": -1.0, "torture": -0.9,
    "shame": -0.8, "shameful": -0.9, "chor": -0.9, "dhokha": -1.0, "jhoot": -0.8,
    "jhooth": -0.8, "lie": -0.7, "lies": -0.7, "emoji_enraged_face": -0.9,
    "emoji_angry_face": -0.9, "emoji_thumbs_down": -0.8, "emoji_crying_face": -0.6,
    "emoji_loudly_crying_face": -0.7, "emoji_face_with_rolling_eyes": -0.6,
    "emoji_face_with_symbols_on_mouth": -1.0, "emoji_pile_of_poo": -0.9,
    "emoji_unamused_face": -0.6, "emoji_face_vomiting": -0.9, "emoji_nauseated_face": -0.8,
    "emoji_middle_finger": -1.0, "emoji_disappointed_face": -0.6, "emoji_weary_face": -0.6,
    "emoji_face_with_steam_from_nose": -0.7, "emoji_pensive_face": -0.4,
    "emoji_slightly_frowning_face": -0.4, "emoji_upside_down_face": -0.3,
}
# remove the accidental 0.0 entry
NEGATIVE = {k: v for k, v in NEGATIVE.items() if v != 0.0}

NEGATORS = {"nahi", "na", "mat", "never", "not", "no", "dont", "didnt", "doesnt",
            "isnt", "wasnt", "cant", "wont", "havent", "hasnt", "couldnt",
            "wouldnt", "bilkul", "kabhi", "without", "bina"}
INTENSIFIERS = {"bahut", "bohot", "very", "so", "too", "extremely", "really",
                "ekdum", "bilkul", "sabse", "zyada", "itna", "itni", "kitna",
                "kitni", "full", "pura", "puri", "totally", "completely", "super",
                "highly", "hadd", "had"}

# Markers that sarcastic Hinglish reviews use a LOT (mock praise / thanks).
SARCASM_MARKERS = {"wah", "wow", "kya_baat", "wah_bhai", "great", "job", "well",
                   "done", "congratulations", "congrats", "thanks", "thank",
                   "shukriya", "dhanyavaad", "dhanyawad", "genius", "brilliant",
                   "best", "amazing", "awesome", "superb", "fantastic", "lol",
                   "haha", "emoji_clapping_hands", "emoji_upside_down_face",
                   "emoji_face_with_rolling_eyes", "emoji_face_with_tears_of_joy",
                   "emoji_rolling_on_the_floor_laughing", "emoji_smiling_face_with_tear",
                   "emoji_slightly_smiling_face", "emoji_winking_face", "obviously",
                   "ofcourse", "of", "course", "expected", "as", "usual", "again",
                   "classic", "typical", "mazaak", "mazak", "joke", "comedy",
                   "sach", "sachme", "sachmein", "really"}


def sentiment_score(tok: str) -> float:
    return POSITIVE.get(tok, 0.0) + NEGATIVE.get(tok, 0.0)
