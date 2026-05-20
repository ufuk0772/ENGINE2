import re
import numpy as np


# ----------------------------
# 1. RULE-BASED SIGNALS
# ----------------------------
SALE_KEYWORDS = [
    "satis", "satış", "sales", "volume", "qty", "quantity", "adet"
]

PRICE_KEYWORDS = [
    "fiyat", "price", "cost", "unitprice", "unit_price"
]

TIME_KEYWORDS = [
    "date", "tarih", "time", "day", "week", "month"
]


# ----------------------------
# 2. SCORE FUNCTION
# ----------------------------
def score_column(col):

    c = col.lower()

    score = {
        "sales": 0,
        "price": 0,
        "time": 0
    }

    # sales signals
    if any(k in c for k in SALE_KEYWORDS):
        score["sales"] += 3

    if re.search(r"(qty|vol|amount|adet)", c):
        score["sales"] += 2

    # price signals
    if any(k in c for k in PRICE_KEYWORDS):
        score["price"] += 3

    if re.search(r"(tl|usd|eur|cost)", c):
        score["price"] += 2

    # time signals
    if any(k in c for k in TIME_KEYWORDS):
        score["time"] += 3

    return score


# ----------------------------
# 3. MAIN MAPPER
# ----------------------------
def build_schema_map(df):

    schema = {}
    product_map = {}

    for col in df.columns:
        s = score_column(col)

        # time column
        if s["time"] >= 3:
            schema["time"] = col
            continue

        # sales
        if s["sales"] >= s["price"] and s["sales"] > 0:
            base = clean_base(col)
            product_map.setdefault(base, {})["q"] = col

        # price
        if s["price"] > s["sales"] and s["price"] > 0:
            base = clean_base(col)
            product_map.setdefault(base, {})["p"] = col

    # fallback enrichment (numeric inference)
    product_map = auto_infer_missing(df, product_map)

    return schema, product_map


# ----------------------------
# 4. BASE NAME CLEANER
# ----------------------------
def clean_base(col):
    c = col.lower()

    for w in SALE_KEYWORDS + PRICE_KEYWORDS:
        c = c.replace(w, "")

    c = re.sub(r"[_\- ]+", "_", c)
    return c.strip("_")


# ----------------------------
# 5. AUTO INFERENCE (IMPORTANT)
# ----------------------------
def auto_infer_missing(df, product_map):

    numeric_cols = df.select_dtypes(include=np.number).columns

    for prod, vals in product_map.items():

        # missing sales
        if "q" not in vals:
            candidates = [c for c in numeric_cols if "sales" in c.lower() or "satis" in c.lower()]
            if candidates:
                vals["q"] = candidates[0]

        # missing price
        if "p" not in vals:
            candidates = [c for c in numeric_cols if "price" in c.lower() or "fiyat" in c.lower()]
            if candidates:
                vals["p"] = candidates[0]

    return product_map