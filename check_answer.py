# check_answer.py — проверка answer.csv перед отправкой
import pandas as pd

# dtype=str, чтобы id не превратились в числа
a = pd.read_csv("answer.csv", dtype=str)
q = pd.read_parquet("data/benchmark_queries.parquet")
items = set(pd.read_parquet("data/benchmark_items.parquet")["item_id"])

print("колонки:", list(a.columns))                                   # ['query_id', 'answer']
print("строк:", len(a), "| ожидается:", len(q))
print("query_id совпадают с бенчмарком:", set(a["query_id"]) == set(q["query_id"]))
print("повторов query_id:", a["query_id"].duplicated().sum())        # ожидается 0

ids = a["answer"].str.split()
print("макс. объявлений в строке:", ids.str.len().max())             # ожидается <= 50
print("повторы внутри строки:", (ids.str.len() != ids.map(set).str.len()).sum())  # ожидается 0
print("все item_id есть в корпусе:", all(i in items for row in ids for i in row))