# diag_loc.py — есть ли центры для локаций запросов бенчмарка
import sys
import pandas as pd
sys.stdout.reconfigure(encoding="utf-8")

train = pd.read_parquet("data/train.parquet")
items = pd.read_parquet("data/benchmark_items.parquet")
q = pd.read_parquet("data/benchmark_queries.parquet")

tr_locs = set(pd.to_numeric(train["search_location_id"], errors="coerce").dropna())
it_locs = set(pd.to_numeric(items["item_location_id"], errors="coerce").dropna())
bl = pd.to_numeric(q["search_location_id"], errors="coerce")

print("Запросы бенчмарка с локацией из train:", bl.isin(tr_locs).mean())
print("Запросы бенчмарка с локацией из корпуса объявлений:", bl.isin(it_locs).mean())
print("Запросы с локацией хоть где-то:", bl.isin(tr_locs | it_locs).mean())