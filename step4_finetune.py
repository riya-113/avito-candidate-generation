# step4_finetune.py — дообучение e5 на парах "запрос -> выбранное объявление" из train
# Метод: контрастное обучение с in-batch negatives (MultipleNegativesRankingLoss):
# в пачке из 128 пар каждый запрос учится быть ближе к своему объявлению, чем к 127 чужим.
import os, random
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sentence_transformers import (SentenceTransformer, SentenceTransformerTrainer,
                                   SentenceTransformerTrainingArguments, losses)
from sentence_transformers.training_args import BatchSamplers

from step2_baseline import query_text, DATA
from step3_hybrid import passage_text

SEED = 42
BASE = "intfloat/multilingual-e5-base"
OUT = "models/e5-avito"
MAX_PAIRS = int(os.environ.get("MAX_PAIRS", 300000))   # сколько пар брать в обучение


def val2_exclusions(train, items, n_val=1000):
    """Те же запросы, что в validate_real (та же выборка с random_state=42).
    Их тексты и верные объявления исключаем из обучения, чтобы валидация была честной.
    ВАЖНО: логика должна совпадать с validate_real в step3_hybrid.py."""
    rows = train[train["item_id"].isin(set(items["item_id"]))]
    qcols = ["search_query", "search_location_id", "search_is_delivery_search",
             "search_infm_params_text", "search_category"]
    groups = rows.groupby(qcols, dropna=False)["item_id"].apply(set).reset_index()
    groups = groups.sample(n=min(n_val, len(groups)), random_state=42).reset_index(drop=True)
    return set(groups["search_query"]), set().union(*groups["item_id"])


def build_pairs():
    """Пары для обучения: anchor = запрос с фильтрами, positive = выбранное объявление."""
    train = pd.read_parquet(DATA + "train.parquet")
    items = pd.read_parquet(DATA + "benchmark_items.parquet")
    val_texts, val_ids = val2_exclusions(train, items)

    tr = train[~train["search_query"].isin(val_texts) & ~train["item_id"].isin(val_ids)]
    # Одна и та же пара "запрос+фильтр -> объявление" из разных городов не нужна дважды
    tr = tr.drop_duplicates(["search_query", "search_infm_params_text", "item_id"])
    tr = tr.sample(n=min(MAX_PAIRS, len(tr)), random_state=SEED).reset_index(drop=True)
    print("Пар для обучения:", len(tr))

    anchors = ["query: " + s for s in query_text(tr)]   # тот же формат, что при поиске
    positives = passage_text(tr)                         # уже с префиксом "passage: "
    return Dataset.from_dict({"anchor": anchors, "positive": positives})


if __name__ == "__main__":   # на Windows обязательно
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    ds = build_pairs()

    model = SentenceTransformer(BASE, device="cuda")
    model.max_seq_length = 192        # короче = быстрее; начало описания всё равно влезает
    loss = losses.MultipleNegativesRankingLoss(model)

    args = SentenceTransformerTrainingArguments(
        output_dir="models/checkpoints",
        num_train_epochs=1,
        per_device_train_batch_size=128,  # больше пачка = больше отрицательных примеров
        learning_rate=2e-5,               # маленький шаг, чтобы не «забыть» исходные знания
        warmup_steps=100,
        bf16=True,                        # RTX 5080 поддерживает bf16: быстрее и экономнее
        batch_sampler=BatchSamplers.NO_DUPLICATES,  # без повторов текста внутри пачки
        dataloader_num_workers=0,         # Windows: без многопроцессности
        logging_steps=100,
        save_strategy="no",
        seed=SEED,
        report_to="none",
    )
    trainer = SentenceTransformerTrainer(model=model, args=args, train_dataset=ds, loss=loss)
    trainer.train()
    model.save(OUT)
    print("Модель сохранена в", OUT)