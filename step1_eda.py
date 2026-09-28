# step1_eda.py - разведочный анализ данных для кандидатогенерации
import pandas as pd

DATA = "data/"

# Читаем parquet. Идентификаторы остаются строками, пока мы не сохраняем в CSV
train = pd.read_parquet(DATA + "train.parquet")
queries = pd.read_parquet(DATA + "benchmark_queries.parquet")
items = pd.read_parquet(DATA + "benchmark_items.parquet")

print("Колонки train:", list(train.columns))
print("Колонки queries:", list(queries.columns))

# 1. Пустоты в ключевых полях (доля пустых/NaN) 
def empty_share(s):
    return (s.isna() | (s.astype(str).str.strip() == "")).mean()

for col in ["search_query", "search_category", "search_infm_params_text"]:
    print(f"train {col}: пустых {empty_share(train[col]):.1%}")
for col in ["item_title_raw", "item_description_raw", "item_infm_params_text"]:
    print(f"items {col}: пустых {empty_share(items[col]):.1%}")

# 2. Длины текстов (в словах): где живёт информация 
for col in ["item_title_raw", "item_description_raw", "item_infm_params_text"]:
    print(f"{col}: медиана слов =", items[col].fillna("").str.split().str.len().median())

# 3. Повторяются ли тексты запросов между train и benchmark 
q_train = set(train["search_query"].str.lower().str.strip())
q_bench = queries["search_query"].str.lower().str.strip()
print("Доля benchmark-запросов, текст которых есть в train:", q_bench.isin(q_train).mean())

# 4. Пересечение объявлений train и корпуса (по заголовку+описанию) 
key = lambda d: d["item_title_raw"].fillna("") + "||" + d["item_description_raw"].fillna("")
print("Доля train-пар, чьё объявление есть в корпусе:", key(train).isin(set(key(items))).mean())

# 5. Сколько объявлений выбирают на один запрос в train 
per_q = train.groupby(["search_query", "search_location_id"]).size()
print("Объявлений на (запрос, локацию), описание:\n", per_q.describe())

# 6. Совпадение категорий: search_category vs item_category_id 
print("Уникальных search_category:", train["search_category"].nunique())
print("Уникальных item_category_id:", items["item_category_id"].nunique())
print(train[["search_category", "item_category_id"]].value_counts().head(10))
# 7. Насколько категория вообще разделяет данные 
print("search_category (benchmark):\n", queries["search_category"].value_counts().head())
print("item_category_id (корпус):\n", items["item_category_id"].value_counts().head())
print("Уникальных microcat в корпусе:", items["item_microcat_id"].nunique())

# 8. Как фильтр запроса соотносится с параметрами выбранного объявления 
has_f = train["search_infm_params_text"].fillna("").str.strip() != ""
print("Доля train-пар с фильтром в запросе:", has_f.mean())
print(train.loc[has_f, ["search_query", "search_infm_params_text", "item_infm_params_text"]]
      .sample(5, random_state=0).to_string())
