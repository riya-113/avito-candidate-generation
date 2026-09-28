# step3_hybrid.py — гибридный поиск: эмбеддинги e5 (GPU) + TF-IDF + бонус за город
import os, re, sys
import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer

# Переиспользуем функции из бейзлайна (без копипасты)
from step2_baseline import analyzer, item_text, query_text, loc_ids, recall_at_k, DATA, K

sys.stdout.reconfigure(encoding="utf-8")
MODEL = os.environ.get("MODEL", "models/e5-avito")   # дообученная модель (итоговое решение)
CACHE_SUFFIX = "" if MODEL == "intfloat/multilingual-e5-base" else "_ft"
RADIUS = float(os.environ.get("RADIUS", 200))   # радиус затухания расстояния, км
POP = float(os.environ.get("POP", 1))          # вес популярности (0 = выключено)
KIND = float(os.environ.get("KIND", 0))   # вес совпадения "Вид услуги" (0 = выключено)
os.makedirs("cache", exist_ok=True)


def kv(params, key):
    """Достаём значение вида 'Вид услуги X' / 'Тип услуги X' из строки параметров."""
    m = re.search(re.escape(key) + r" (.+?)(?= (?:Тип услуги|Вид услуги|Место оказания|Тип стоимости)|$)", params)
    return m.group(1) if m else ""


def passage_text(df):
    """Короткий текст объявления для эмбеддера: заголовок + вид/тип услуги + начало описания.
    Полные параметры слишком длинные и забивают лимит токенов адресами и марками авто."""
    out = []
    for t, p, d in zip(df["item_title_raw"].fillna(""), df["item_infm_params_text"].fillna(""),
                       df["item_description_raw"].fillna("")):
        out.append("passage: " + t + ". " + kv(p, "Вид услуги") + ". " + kv(p, "Тип услуги") + ". " + d[:500])
    return out


def query_prefixed(df):
    return ["query: " + s for s in query_text(df)]


_model = None
def encode(texts, tag):
    """Считает эмбеддинги на GPU и кэширует на диск (повторный запуск быстрый)."""
    global _model
    path = f"cache/{tag}{CACHE_SUFFIX}.npy"
    if os.path.exists(path):
        return np.load(path)
    if _model is None:
        _model = SentenceTransformer(MODEL, device="cuda")
        _model.half()                 # fp16: вдвое быстрее и меньше памяти
        _model.max_seq_length = 256
    emb = _model.encode(texts, batch_size=128, normalize_embeddings=True,
                        convert_to_numpy=True, show_progress_bar=True).astype(np.float16)
    np.save(path, emb)
    return emb


def zscore(S):
    """Приводим оценки каждого запроса к одной шкале (среднее 0, разброс 1), чтобы их можно было складывать."""
    return (S - S.mean(1, keepdim=True)) / (S.std(1, keepdim=True) + 1e-6)

def location_centers(train):
    """Центр каждой локации запроса = медиана координат выбранных там объявлений (только train)."""
    g = train.groupby("search_location_id")[["item_latitude", "item_longitude"]].median()
    g.index = pd.to_numeric(g.index, errors="coerce")
    return g


def query_coords(queries, centers):
    """Координаты центра для каждого запроса. Если локации нет в train, ставим NaN."""
    loc = pd.to_numeric(queries["search_location_id"], errors="coerce")
    lat = loc.map(centers["item_latitude"]).to_numpy(dtype=np.float32)
    lon = loc.map(centers["item_longitude"]).to_numpy(dtype=np.float32)
    return lat, lon


def dist_km(lat1, lon1, lat2, lon2):
    """Расстояние в км между точками (упрощённо, для сравнения хватает). Все аргументы - тензоры."""
    dlat = (lat1 - lat2) * 111.0
    dlon = (lon1 - lon2) * 111.0 * torch.cos(torch.deg2rad((lat1 + lat2) / 2))
    return torch.sqrt(dlat ** 2 + dlon ** 2)


def retrieve(queries, corpus, D_emb, Q_emb, configs, centers, chunk=256):
    """configs = список (w_dense, loc_bonus). Возвращает {config: [список item_id на запрос]}."""
    vec = TfidfVectorizer(analyzer=analyzer, sublinear_tf=True, min_df=2, dtype=np.float32)
    D = vec.fit_transform(item_text(corpus)).T.tocsr()
    Q = vec.transform(query_text(queries))
    ids = corpus["item_id"].to_numpy()
    dev = "cuda"
    D_t = torch.from_numpy(D_emb).to(dev)                      # эмбеддинги корпуса на GPU
    item_loc = torch.from_numpy(loc_ids(corpus["item_location_id"])).to(dev)
    q_loc = torch.from_numpy(loc_ids(queries["search_location_id"])).to(dev)
    i_lat = torch.from_numpy(corpus["item_latitude"].fillna(0).to_numpy(np.float32)).to(dev)
    i_lon = torch.from_numpy(corpus["item_longitude"].fillna(0).to_numpy(np.float32)).to(dev)
    reviews = corpus["item_rating_reviews_count"].fillna(0).astype(float).to_numpy()
    i_pop = torch.from_numpy((np.log1p(reviews) / np.log1p(reviews.max())).astype(np.float32)).to(dev)
    i_kind = pd.Series([kv(p, "Вид услуги") for p in corpus["item_infm_params_text"].fillna("")])
    q_kind = pd.Series([kv(p, "Вид услуги") for p in queries["search_infm_params_text"].fillna("")])
    codes = {v: i + 1 for i, v in enumerate(pd.unique(pd.concat([i_kind, q_kind])))}
    codes[""] = 0                                    # пустое значение не считается совпадением
    i_kc = torch.from_numpy(i_kind.map(codes).to_numpy(np.int64)).to(dev)
    q_kc = torch.from_numpy(q_kind.map(codes).to_numpy(np.int64)).to(dev)
    qlat, qlon = query_coords(queries, centers)
    q_lat = torch.from_numpy(np.nan_to_num(qlat)).to(dev)
    q_lon = torch.from_numpy(np.nan_to_num(qlon)).to(dev)
    q_has = torch.from_numpy(~np.isnan(qlat)).float().to(dev)   # 1, если центр известен
    res = {c: [] for c in configs}
    for s in range(0, Q.shape[0], chunk):
        e = min(s + chunk, Q.shape[0])
        lex = zscore(torch.from_numpy((Q[s:e] @ D).toarray()).to(dev))   # оценка по словам
        sem = zscore(torch.from_numpy(Q_emb[s:e]).to(dev) @ D_t.T)       # оценка по смыслу
        same_city = (item_loc[None, :] == q_loc[s:e, None]).float()
        kind_match = ((i_kc[None, :] == q_kc[s:e, None]) & (q_kc[s:e, None] > 0)).float()
        d = dist_km(q_lat[s:e, None], q_lon[s:e, None], i_lat[None, :], i_lon[None, :])
        near = torch.exp(-d / RADIUS) * q_has[s:e, None]       # 1 рядом, ~0 за 200+ км
        for (w, lb, db) in configs:
            S = w * sem.float() + (1 - w) * lex + lb * same_city + db * near + POP * i_pop[None, :] + KIND * kind_match
            top = torch.topk(S, K, dim=1).indices.cpu().numpy()
            res[(w, lb, db)].extend(list(ids[row]) for row in top)
        print(f"  {e}/{Q.shape[0]}")
    return res


def validate(n_val=1000):
    """Честная проверка: запросы из train, их ответы спрятаны среди чужих объявлений."""
    train = pd.read_parquet(DATA + "train.parquet")
    items = pd.read_parquet(DATA + "benchmark_items.parquet")
    texts = train["search_query"].drop_duplicates().sample(frac=1, random_state=42)
    val_rows = train[train["search_query"].isin(set(texts.iloc[:n_val]))]
    qcols = ["search_query", "search_location_id", "search_is_delivery_search",
             "search_infm_params_text", "search_category"]
    groups = val_rows.groupby(qcols, dropna=False)["item_id"].apply(set).reset_index()
    groups = groups.sample(n=min(n_val, len(groups)), random_state=42).reset_index(drop=True)
    relevant = groups["item_id"].tolist()

    val_items = val_rows[list(items.columns)].drop_duplicates("item_id")
    val_items = val_items[~val_items["item_id"].isin(set(items["item_id"]))].reset_index(drop=True)
    corpus = pd.concat([items, val_items]).reset_index(drop=True)
    D_emb = np.vstack([encode(passage_text(items), "items"), encode(passage_text(val_items), "val_items")])
    Q_emb = encode(query_prefixed(groups), "val_queries")
    print("Корпус:", len(corpus), "| запросов:", len(groups))
    val_item_ids = set().union(*relevant)   # все верные ответы валидации
    train_fit = train[~train["search_query"].isin(set(texts.iloc[:n_val])) & ~train["item_id"].isin(val_item_ids)]
    centers = location_centers(train_fit)
    configs = [(0.95, lb, db) for lb in (1.0, 2.0, 4.0) for db in (2.0, 4.0, 6.0)]
    res = retrieve(groups, corpus, D_emb, Q_emb, configs, centers)
    for c in configs:
        print(f"w=0.85 loc={c[1]} dist={c[2]}: Recall@{K} = {recall_at_k(res[c], relevant):.4f}")


def submit(w, lb, db=0.0):
    queries = pd.read_parquet(DATA + "benchmark_queries.parquet")
    items = pd.read_parquet(DATA + "benchmark_items.parquet")
    train = pd.read_parquet(DATA + "train.parquet")
    centers = location_centers(train)          # центры локаций по всему train
    D_emb = encode(passage_text(items), "items")
    Q_emb = encode(query_prefixed(queries), "bench_queries")
    cfg = (w, lb, db)
    preds = retrieve(queries, items, D_emb, Q_emb, [cfg], centers)[cfg]
    ans = pd.DataFrame({"query_id": queries["query_id"].astype(str), "answer": [" ".join(p) for p in preds]})
    ans.to_csv("answer.csv", index=False)
    print("answer.csv сохранён, строк:", len(ans))

def validate_real(n_val=1000):
    """Честная проверка: запросы из train, чьи верные объявления УЖЕ лежат в benchmark_items.
    Корпус = только benchmark_items, ничего не добавляем (как в настоящем бенчмарке)."""
    train = pd.read_parquet(DATA + "train.parquet")
    items = pd.read_parquet(DATA + "benchmark_items.parquet")
    rows = train[train["item_id"].isin(set(items["item_id"]))]
    qcols = ["search_query", "search_location_id", "search_is_delivery_search",
             "search_infm_params_text", "search_category"]
    groups = rows.groupby(qcols, dropna=False)["item_id"].apply(set).reset_index()
    groups = groups.sample(n=min(n_val, len(groups)), random_state=42).reset_index(drop=True)
    relevant = groups["item_id"].tolist()

    # Центры локаций считаем без валидационных запросов и без их верных объявлений (нет утечки)
    val_texts = set(groups["search_query"])
    val_ids = set().union(*relevant)
    train_fit = train[~train["search_query"].isin(val_texts) & ~train["item_id"].isin(val_ids)]
    centers = location_centers(train_fit)

    D_emb = encode(passage_text(items), "items")                 # из кэша
    Q_emb = encode(query_prefixed(groups), "val2_queries")       # новый кэш
    print("Корпус:", len(items), "| запросов:", len(groups))
    configs = [(0.95, lb, db) for lb in (1.0, 2.0, 4.0) for db in (2.0, 4.0, 6.0)]
    res = retrieve(groups, items, D_emb, Q_emb, configs, centers)
    for c in configs:
        print(f"w={c[0]} loc={c[1]} dist={c[2]}: Recall@{K} = {recall_at_k(res[c], relevant):.4f}")
    return groups, relevant, res[configs[0]], items, centers
def analyze_errors():
    """Разбор промахов на val2: какие верные объявления не попали в топ-50 и чем они отличаются."""
    import math
    groups, relevant, preds, items, centers = validate_real()
    it = items.set_index("item_id")
    qlat, qlon = query_coords(groups, centers)
    item_kinds = set(kv(p, "Вид услуги") for p in items["item_infm_params_text"].fillna(""))

    rows = []
    for i, (g, rel, p) in enumerate(zip(groups.itertuples(index=False), relevant, preds)):
        found = set(p)
        qk = kv(g.search_infm_params_text or "", "Вид услуги")
        for iid in rel:
            r = it.loc[iid]
            ik = kv(r.item_infm_params_text or "", "Вид услуги")
            if np.isnan(qlat[i]):
                d = np.nan
            else:
                la, lo = float(r.item_latitude), float(r.item_longitude)
                dx = (lo - qlon[i]) * 111 * math.cos(math.radians(la))
                d = math.hypot((la - qlat[i]) * 111, dx)
            rows.append(dict(
                hit=iid in found,
                same_loc=str(r.item_location_id) == str(g.search_location_id),
                dist=d,
                delivery=int(g.search_is_delivery_search),
                has_kind=qk != "",
                kind_parsed_ok=(qk in item_kinds) if qk else None,  # распознали ли значение фильтра
                kind_match=(qk == ik) if qk else None,
                query=g.search_query, title=r.item_title_raw))
    df = pd.DataFrame(rows)

    print("\nОбщий recall по объявлениям:", round(df["hit"].mean(), 4))
    for col in ["same_loc", "delivery", "has_kind", "kind_parsed_ok", "kind_match"]:
        print(f"\n--- recall по {col} ---")
        print(df.groupby(col, dropna=False)["hit"].agg(["mean", "size"]).round(3))
    df["dist_bin"] = pd.cut(df["dist"], [-1, 10, 50, 200, 1000, 1e5])
    print("\n--- recall по расстоянию (км) ---")
    print(df.groupby("dist_bin", observed=True)["hit"].agg(["mean", "size"]).round(3))

    bench = pd.read_parquet(DATA + "benchmark_queries.parquet")
    print("\nДоля запросов с доставкой: val2 =", round(groups["search_is_delivery_search"].astype(int).mean(), 3),
          "| бенчмарк =", round(bench["search_is_delivery_search"].astype(int).mean(), 3))

    print("\n--- 15 примеров промахов ---")
    print(df.loc[~df["hit"], ["query", "title", "same_loc", "dist", "kind_match"]]
          .head(15).to_string(max_colwidth=40))
if __name__ == "__main__":
    if sys.argv[1] == "val":
        validate()
    elif sys.argv[1] == "val2":
        validate_real()
    elif sys.argv[1] == "err":
        analyze_errors()
    else:
        submit(float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]) if len(sys.argv) > 4 else 0.0)