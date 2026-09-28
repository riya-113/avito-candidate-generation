# step1b_checks.py — проверки, от которых зависит архитектура
import re
import sys
import pandas as pd

# Печатаем в UTF-8, чтобы кириллица не превращалась в кракозябры
sys.stdout.reconfigure(encoding="utf-8")

train = pd.read_parquet("data/train.parquet")
items = pd.read_parquet("data/benchmark_items.parquet")

#  A. Пересечение train и корпуса по item_id (точнее, чем по тексту) 
in_corpus = train["item_id"].isin(set(items["item_id"]))
print("Доля train-пар, чей item_id есть в корпусе:", in_corpus.mean())
print("Уникальных item_id в train:", train["item_id"].nunique())

#  B. Совпадает ли локация запроса и локация выбранного объявления 
same_loc = train["search_location_id"].astype(str) == train["item_location_id"].astype(str)
print("Локация запроса == локация объявления:", same_loc.mean())

#  C. Фильтр 'Вид услуги' / 'Тип услуги' из запроса внутри параметров объявления 
def check_key(key, stop_keys):
    """Достаём значение фильтра по ключу и проверяем, есть ли 'ключ значение' в параметрах объявления."""
    # Значение идёт до следующего известного ключа или до конца строки
    stop = "|".join(map(re.escape, stop_keys))
    pat = re.compile(re.escape(key) + r" (.+?)(?= (?:" + stop + r")|\s*$)")
    q = train["search_infm_params_text"].fillna("")
    vals = q.str.extract(pat, expand=False)
    has = vals.notna()
    hit = pd.Series(
        [(key + " " + v) in p for v, p in zip(vals[has], train.loc[has, "item_infm_params_text"].fillna(""))],
        index=vals[has].index,
    )
    print(f"\n[{key}] строк с фильтром: {has.mean():.1%} от train; совпало: {hit.mean():.2%}")
    bad = train.loc[hit[~hit].index, ["search_infm_params_text", "item_infm_params_text"]].head(8)
    for _, r in bad.iterrows():
        print("  ЗАПРОС:", r["search_infm_params_text"])
        print("  ОБЪЯВЛ:", r["item_infm_params_text"][:150])

check_key("Вид услуги", ["Тип услуги", "Онлайн-запись", "Рейтинг"])
check_key("Тип услуги", ["Вид услуги", "Онлайн-запись", "Рейтинг"])
