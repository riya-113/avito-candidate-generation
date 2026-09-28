# diag_geo.py — одинаково ли устроены координаты в train и в корпусе
import sys
import numpy as np
import pandas as pd
sys.stdout.reconfigure(encoding="utf-8")

train = pd.read_parquet("data/train.parquet").drop_duplicates("item_id")
items = pd.read_parquet("data/benchmark_items.parquet")

def stats(name, df):
    lat = df["item_latitude"]
    print(f"[{name}] объявлений: {len(df)}")
    print(f"[{name}] пустые координаты: {lat.isna().mean():.3%}, нули: {(lat == 0).mean():.3%}")
    pairs = df[["item_latitude", "item_longitude"]].drop_duplicates().shape[0]
    print(f"[{name}] уникальных пар координат на объявление: {pairs / len(df):.3f}")

def own_dist(name, df):
    """Расстояние (км) от объявления до медианы координат его же локации."""
    df = df.dropna(subset=["item_latitude", "item_longitude"]).copy()
    df["loc"] = pd.to_numeric(df["item_location_id"], errors="coerce")
    med = df.groupby("loc")[["item_latitude", "item_longitude"]].transform("median")
    dlat = (df["item_latitude"] - med["item_latitude"]) * 111
    dlon = (df["item_longitude"] - med["item_longitude"]) * 111 * np.cos(np.deg2rad(df["item_latitude"]))
    d = np.sqrt(dlat ** 2 + dlon ** 2)
    print(f"[{name}] расстояние до центра своей локации, км, квантили 50/75/90/99%:",
          [round(float(d.quantile(q)), 1) for q in (0.5, 0.75, 0.9, 0.99)])

stats("train", train)
stats("corpus", items)
own_dist("train", train)
own_dist("corpus", items)