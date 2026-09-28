# step2_baseline.py — лексический бейзлайн (TF-IDF) + валидация + answer.csv
import re, sys
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

sys.stdout.reconfigure(encoding="utf-8")
DATA = "data/"
K = 50            # сколько кандидатов возвращаем
LOC_BONUS = 0.05  # прибавка к оценке, если город запроса == город объявления


def analyzer(text):
    """Разбиваем текст на слова и обрезаем каждое до 5 букв (грубая лемматизация)."""
    return [w[:5] for w in re.findall(r"\w+", text.lower())]


def item_text(df):
    """Текст объявления: заголовок (3 раза = больший вес) + параметры + начало описания."""
    t = df["item_title_raw"].fillna("")
    p = df["item_infm_params_text"].fillna("").str.slice(0, 500)
    d = df["item_description_raw"].fillna("").str.slice(0, 300)
    return t + " " + t + " " + t + " " + p + " " + d


def query_text(df):
    """Текст запроса: сам запрос + его фильтры (Вид/Тип услуги совпадают с параметрами объявления)."""
    return df["search_query"].fillna("") + " " + df["search_infm_params_text"].fillna("")


def loc_ids(series):
    """Локации -> целые числа (для быстрого сравнения)."""
    return pd.to_numeric(series, errors="coerce").fillna(-1).astype(np.int64).to_numpy()


def retrieve(queries, items, k=K, loc_bonus=LOC_BONUS, chunk=256):
    """Для каждого запроса возвращает список из k item_id с наибольшей оценкой."""
    vec = TfidfVectorizer(analyzer=analyzer, sublinear_tf=True, min_df=2, dtype=np.float32)
    D = vec.fit_transform(item_text(items)).T.tocsr()   # матрица слова x объявления
    Q = vec.transform(query_text(queries))               # матрица запросы x слова
    item_ids = items["item_id"].to_numpy()
    item_loc = loc_ids(items["item_location_id"])
    q_loc = loc_ids(queries["search_location_id"])

    result = []
    for s in range(0, Q.shape[0], chunk):
        S = (Q[s:s + chunk] @ D).toarray()               # похожесть запрос-объявление
        S += loc_bonus * (item_loc[None, :] == q_loc[s:s + chunk, None])  # бонус за город
        top = np.argpartition(-S, k, axis=1)[:, :k]      # индексы k лучших (без сортировки)
        for i in range(top.shape[0]):
            order = top[i][np.argsort(-S[i, top[i]])]    # сортируем внутри топ-k
            result.append(list(item_ids[order]))
        print(f"  обработано {min(s + chunk, Q.shape[0])}/{Q.shape[0]}")
    return result


def recall_at_k(preds, relevant):
    """Средняя доля найденных релевантных объявлений."""
    return float(np.mean([len(set(p) & r) / len(r) for p, r in zip(preds, relevant)]))


def validate(n_val=1000):
    """Честная проверка: берём запросы из train, прячем их ответы среди чужих объявлений."""
    train = pd.read_parquet(DATA + "train.parquet")
    items = pd.read_parquet(DATA + "benchmark_items.parquet")

    # Делим по ТЕКСТУ запроса, чтобы один текст не попал в обе части
    texts = train["search_query"].drop_duplicates().sample(frac=1, random_state=42)
    val_texts = set(texts.iloc[:n_val])
    val_rows = train[train["search_query"].isin(val_texts)]

    # Один валидационный "запрос" = уникальная комбинация признаков запроса
    qcols = ["search_query", "search_location_id", "search_is_delivery_search",
             "search_infm_params_text", "search_category"]
    groups = val_rows.groupby(qcols, dropna=False)["item_id"].apply(set).reset_index()
    groups = groups.sample(n=min(n_val, len(groups)), random_state=42).reset_index(drop=True)
    relevant = groups["item_id"].tolist()

    # Корпус для проверки = чужие объявления (benchmark_items) + верные ответы валидации
    icols = [c for c in items.columns]
    val_items = val_rows[icols].drop_duplicates("item_id")
    corpus = pd.concat([items, val_items]).drop_duplicates("item_id").reset_index(drop=True)
    print("Размер корпуса для валидации:", len(corpus), "| запросов:", len(groups))

    for bonus in [0.0, LOC_BONUS]:
        preds = retrieve(groups, corpus, loc_bonus=bonus)
        print(f"Recall@{K} при loc_bonus={bonus}: {recall_at_k(preds, relevant):.4f}")


def submit():
    """Готовим answer.csv для отправки."""
    queries = pd.read_parquet(DATA + "benchmark_queries.parquet")
    items = pd.read_parquet(DATA + "benchmark_items.parquet")
    preds = retrieve(queries, items)
    answer = pd.DataFrame({
        "query_id": queries["query_id"].astype(str),
        "answer": [" ".join(p) for p in preds],
    })
    answer.to_csv("answer.csv", index=False)
    # Проверки формата
    assert len(answer) == queries["query_id"].nunique()
    assert answer["answer"].str.split().str.len().max() <= K
    print("answer.csv сохранён, строк:", len(answer))


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "val"
    validate() if mode == "val" else submit()