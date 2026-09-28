# compare.py — совпадают ли два ответа (порядок внутри строки не важен)
import sys
import pandas as pd

a = pd.read_csv(sys.argv[1], dtype=str)
b = pd.read_csv(sys.argv[2], dtype=str)
m = a.merge(b, on="query_id")
same = (m["answer_x"].str.split().map(set) == m["answer_y"].str.split().map(set)).mean()
print(f"{sys.argv[1]} vs {sys.argv[2]}: одинаковых строк {same:.1%}")