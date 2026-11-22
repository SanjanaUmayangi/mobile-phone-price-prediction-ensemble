"""Cleaning + feature engineering for the Sri Lankan mobile phone listings.

Input : data/mobile_phones_raw.csv  (columns: id, title, brand, model, price, price_raw,
        condition, storage, ram, location, description, detail_*, seller_name,
        posted_date, ad_url, is_member, is_featured)
Output: a cleaned dataframe with the columns
        brand, model, condition, storage_gb, ram_gb, location, source, price (+ helper columns)

The raw CSV is never overwritten.
"""
import re

import numpy as np
import pandas as pd

from . import config

# ----------------------------------------------------------------------------
# 1. RAM / storage extraction
# ----------------------------------------------------------------------------
_SIZE = re.compile(r"(\d+(?:\.\d+)?)\s*(tb|gb)", re.I)
_PAIR = re.compile(r"(?<![\d.])(\d{1,2})\s*[/+]\s*(\d{2,4})\s*(?:gb)?", re.I)  # 8/256, 6+128
_RAM_VALUES = {1, 2, 3, 4, 6, 8, 12, 16, 18, 24}


def parse_specs(*texts):
    """Return (ram_gb, storage_gb) from any number of text fields.

    The CSV 'storage' / 'ram' columns are sometimes swapped or hold the wrong number
    (e.g. '8GB' in storage for an 8GB/256GB phone), so every text field is read and the
    numbers are classified by keyword (RAM / ROM) first and by size second.
    """
    text = " ".join(str(t) for t in texts if isinstance(t, str) and t.strip()).lower()
    if not text:
        return np.nan, np.nan

    ram_explicit, sto_explicit, values = None, None, []
    for m in _SIZE.finditer(text):
        v = float(m.group(1)) * (1024 if m.group(2) == "tb" else 1)
        before = text[max(0, m.start() - 6): m.start()]
        after = text[m.end(): m.end() + 6]
        if "ram" in after or "ram" in before:
            ram_explicit = v if ram_explicit is None else ram_explicit
        elif any(k in after or k in before for k in ("rom", "storage", "ssd", "internal")):
            sto_explicit = v if sto_explicit is None else sto_explicit
        values.append(v)

    # '8/256' style pairs where the first number has no unit
    pair = _PAIR.search(text)
    if pair:
        a, b = float(pair.group(1)), float(pair.group(2))
        if a in _RAM_VALUES and b >= 16:
            ram_explicit = ram_explicit if ram_explicit is not None else a
            sto_explicit = sto_explicit if sto_explicit is not None else b
            values += [a, b]

    values = sorted(set(values))
    big = [v for v in values if v >= 32]
    small = [v for v in values if v < 32]

    ram, sto = ram_explicit, sto_explicit
    if sto is None:
        if big:
            sto = max(big)
        elif len(small) >= 2:                      # e.g. 2GB / 16GB on an old phone
            sto = max(small)
        elif len(small) == 1 and small[0] == 16 and ram is None:
            sto = 16.0
    if ram is None:
        cand = [v for v in small if v in _RAM_VALUES and v != sto]
        if cand:
            ram = max(cand)
    if ram is not None and (ram not in _RAM_VALUES):
        ram = np.nan
    if sto is not None and sto < 4:
        sto = np.nan
    return (np.nan if ram is None else ram), (np.nan if sto is None else sto)


# ----------------------------------------------------------------------------
# 2. Model-name cleaning
# ----------------------------------------------------------------------------
_COLOURS = (
    "black white blue red green gold silver grey gray purple pink yellow orange titanium "
    "natural midnight starlight graphite violet lavender cream brown teal mint cyan bronze "
    "coral lilac ocean sierra alpine space deep desert"
).split()
_NOISE = set(
    "dual sim brand new used global version official warranty pta with box original "
    "unlocked mobile phone smartphone ram rom storage for sale only full set".split()
) | set(_COLOURS)


def clean_model_name(model, title, brand):
    """'Galaxy A56 5G 8GB 256GB' -> 'galaxy a56 5g' (max 4 words)."""
    s = model if isinstance(model, str) and model.strip() else (title if isinstance(title, str) else "")
    s = s.lower()
    s = re.sub(r"\(.*?\)", " ", s)                       # (Brand New), (Used)
    s = _SIZE.sub(" ", s)                                # 8GB, 256GB, 1TB
    s = re.sub(r"(?<![\d.])\d{1,2}\s*[/+]\s*\d{2,4}\b", " ", s)
    s = re.sub(r"[^a-z0-9+ ]", " ", s)
    if isinstance(brand, str) and brand.strip():
        s = re.sub(rf"^\s*{re.escape(brand.lower())}\b", " ", s)
    words = [w for w in s.split() if w not in _NOISE and w != "+"]
    return " ".join(words[:4]) if words else "unknown"


# ----------------------------------------------------------------------------
# 3. Small helpers
# ----------------------------------------------------------------------------
def normalise_condition(cond, title=""):
    for text in (cond, title):
        if isinstance(text, str):
            t = text.lower()
            if "new" in t:
                return "Brand New"
            if "used" in t or "second" in t:
                return "Used"
    return "Unknown"


def derive_source(df, blank_url_source=None):
    """ikman / Dialog from the ad_url (or an existing 'source' column). Never guesses silently."""
    if "source" in df.columns and df["source"].notna().any():
        return df["source"].astype(str).str.strip().replace({"nan": "Unknown"})
    url = df.get("ad_url", pd.Series([None] * len(df), index=df.index)).astype(str).str.lower()
    src = pd.Series("Unknown", index=df.index)
    src[url.str.contains("ikman.lk", na=False)] = "ikman"
    src[url.str.contains("dialog", na=False)] = "Dialog"
    if blank_url_source:
        blank = df.get("ad_url", pd.Series(index=df.index)).isna() | (url.isin(["", "nan", "none"]))
        src[blank & (src == "Unknown")] = blank_url_source
    return src


def parse_price(series):
    s = series.astype(str).str.replace(r"[^\d.]", "", regex=True)
    return pd.to_numeric(s, errors="coerce")


def flag_suspicious_prices(df, low=0.4, high=2.5, min_group=5, abs_floor=3000):
    """Flag (not delete) prices far from the median of the same brand+model+condition group."""
    key = df["model"] + "|" + df["condition"]
    med = df.groupby(key)["price"].transform("median")
    size = df.groupby(key)["price"].transform("size")
    ratio = df["price"] / med
    flag = ((size >= min_group) & ((ratio < low) | (ratio > high))) | (df["price"] < abs_floor)
    return flag.fillna(False)


# ----------------------------------------------------------------------------
# 4. Full cleaning pipeline
# ----------------------------------------------------------------------------
def clean_dataset(raw, blank_url_source=None, drop_suspicious=False, verbose=True):
    """Return (clean_df, report_df). report_df lists how many rows each step removed."""
    steps = []

    def log(name, df):
        steps.append({"step": name, "rows_after": len(df)})

    df = raw.copy()
    log("raw file", df)

    # price -> numeric (fall back to price_raw)
    price = pd.to_numeric(df["price"], errors="coerce") if "price" in df else pd.Series(np.nan, index=df.index)
    if "price_raw" in df:
        price = price.fillna(parse_price(df["price_raw"]))
    df["price"] = price
    df = df.dropna(subset=["price"])
    log("drop missing price", df)
    df = df[df["price"] > 0]
    log("drop price <= 0", df)

    # source
    df["source"] = derive_source(df, blank_url_source)
    n_unknown = int((df["source"] == "Unknown").sum())
    if verbose and n_unknown:
        print(f"WARNING: {n_unknown} rows have source = 'Unknown' (blank / unrecognised ad_url). "
              "Inspect them in notebook 01 and set config.BLANK_URL_SOURCE if appropriate.")

    # exact duplicates (same advert scraped twice)
    if "ad_url" in df:
        has_url = df["ad_url"].notna() & (df["ad_url"].astype(str).str.strip() != "")
        dup_url = df.duplicated(subset=["ad_url"], keep="first") & has_url
        df = df[~dup_url]
    df = df.drop_duplicates(subset=[c for c in ["title", "price", "location", "source"] if c in df])
    log("drop duplicate listings", df)

    # specs
    for c in ["title", "model", "storage", "ram", "detail_model"]:
        if c not in df:
            df[c] = np.nan
    specs = [parse_specs(t, m, s, r, dm) for t, m, s, r, dm in
             zip(df["title"], df["model"], df["storage"], df["ram"], df["detail_model"])]
    df["ram_gb"] = [s[0] for s in specs]
    df["storage_gb"] = [s[1] for s in specs]

    # categoricals
    brand = df["brand"].where(df["brand"].notna() & (df["brand"].astype(str).str.strip() != ""),
                              df.get("detail_brand"))
    brand = brand.fillna(df["title"].astype(str).str.split().str[0])
    df["brand"] = brand.astype(str).str.strip().str.title().replace({"Nan": "Unknown", "Iphone": "Apple"})
    df["condition"] = [normalise_condition(c, t) for c, t in
                       zip(df.get("condition", pd.Series(index=df.index)), df["title"])]
    m_clean = [clean_model_name(m, t, b) for m, t, b in zip(df["model"], df["title"], df["brand"])]
    df["model"] = (df["brand"].str.lower() + " " + pd.Series(m_clean, index=df.index)).str.strip()
    loc = df.get("location", pd.Series("Unknown", index=df.index))
    df["location"] = loc.fillna("Unknown").astype(str).str.strip().str.title().replace({"": "Unknown", "Nan": "Unknown"})

    # suspicious prices
    df["suspicious_price"] = flag_suspicious_prices(df)
    if verbose:
        print(f"{int(df['suspicious_price'].sum())} suspicious prices flagged (kept unless drop_suspicious=True).")
    if drop_suspicious:
        df = df[~df["suspicious_price"]]
        log("drop suspicious prices", df)

    keep = ["id", "title", "brand", "model", "condition", "storage_gb", "ram_gb",
            "location", "source", "price", "suspicious_price", "ad_url"]
    df = df[[c for c in keep if c in df.columns]].reset_index(drop=True)
    log("final clean dataset", df)
    return df, pd.DataFrame(steps)


# ----------------------------------------------------------------------------
# 5. Split (70 / 15 / 15) saved to disk so every notebook uses identical rows
# ----------------------------------------------------------------------------
def make_splits(clean, save=True):
    from sklearn.model_selection import train_test_split

    train, temp = train_test_split(clean, test_size=config.TEST_SIZE + config.VAL_SIZE,
                                   random_state=config.RANDOM_STATE)
    val, test = train_test_split(temp, test_size=config.TEST_SIZE / (config.TEST_SIZE + config.VAL_SIZE),
                                 random_state=config.RANDOM_STATE)
    splits = {"train": train, "val": val, "test": test}
    if save:
        config.SPLIT_DIR.mkdir(parents=True, exist_ok=True)
        for k, v in splits.items():
            v.to_csv(config.SPLIT_DIR / f"{k}.csv", index=False)
    return splits


def load_splits():
    return {k: pd.read_csv(config.SPLIT_DIR / f"{k}.csv") for k in ["train", "val", "test"]}


def xy(df):
    return df[config.FEATURES], df[config.TARGET]


def run_preprocessing(raw_path=None, blank_url_source=None, drop_suspicious=None):
    """Raw CSV -> clean CSV -> splits. Returns (clean, report, splits)."""
    raw = pd.read_csv(raw_path or config.RAW_CSV)
    clean, report = clean_dataset(
        raw,
        blank_url_source=blank_url_source if blank_url_source is not None else config.BLANK_URL_SOURCE,
        drop_suspicious=config.DROP_SUSPICIOUS if drop_suspicious is None else drop_suspicious,
    )
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    clean.to_csv(config.CLEAN_CSV, index=False)
    return clean, report, make_splits(clean)
