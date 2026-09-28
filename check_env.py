# check_env.py — проверка окружения, GPU и целостности данных
import torch
import pandas as pd

# --- 1. Проверка GPU ---
print("torch:", torch.__version__)
print("CUDA доступна:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"VRAM: {vram:.1f} ГБ")

# --- 2. Загрузка данных ---
# В Parquet строки хранятся как строки, поэтому id не портятся.
# Опасность появляется при записи/чтении CSV — учтём это в самом конце.
train = pd.read_parquet("data/train.parquet")
queries = pd.read_parquet("data/benchmark_queries.parquet")
items = pd.read_parquet("data/benchmark_items.parquet")

print("train:", train.shape)
print("queries:", queries.shape)
print("items:", items.shape)

# --- 3. Проверка id: тип, длина 16, нижний регистр ---
for name, s in [("query_id", queries["query_id"]), ("item_id", items["item_id"])]:
    print(name, "dtype:", s.dtype,
          "| все длины == 16:", (s.str.len() == 16).all(),
          "| уникальны:", s.is_unique)

# --- 4. Быстрый взгляд на колонки ---
print(queries.head(3).T)
print(items.head(3).T)