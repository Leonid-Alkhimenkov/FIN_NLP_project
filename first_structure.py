import os
import re
import numpy as np
import pandas as pd
import torch
import regex as re


from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from transformers import (
XLMRobertaTokenizer,
XLMRobertaForSequenceClassification,
Trainer,
TrainingArguments,
BertTokenizer,
BertForSequenceClassification
)
from torch.utils.data import Dataset


os.environ["WANDB_DISABLED"] = "true"


class FinancialDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts.reset_index(drop=True)
        self.labels = labels.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long)
        }

def basic_preprocessing(text: str) -> str:
    """
    Базовая функция предобработки для многоязычного текста:
    1) Приведение к нижнему регистру
    2) Удаление всего, что не является буквой, цифрой, пунктуацией или пробельным символом (в Unicode)
    3) Удаление лишних пробелов
    Примечание: \p{L}захватываетвсебуквы, вт.ч.японские(漢字, カタカナ, ひらがな),
    \p{N} — любыецифры, \p{P} — пунктуацию, \p{Z} — любые пробельные символы(включая неразрывные).
    """
    text = text.lower()
    text = re.sub(r"[^\p{L}\p{N}\p{P}\p{Z}]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text



def compute_metrics(eval_pred):
    """Вычисляет f1_micro при мультиклассовой классификации."""
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)
    f1 = f1_score(labels, preds, average="micro")
    return {"f1_micro": f1}

if __name__ == "__main__":
    # Путь к данным (train.json, val.json) по умолчанию в этой же папке
    train_path = Path(__file__).parent / "0-all-languages-lowlevel/train.json"
    val_path = Path(__file__).parent / "0-all-languages-lowlevel/val.json"
    # train_path = Path(__file__).parent / "1-only-english-lowlevel/train.json"
    # val_path = Path(__file__).parent / "1-only-english-lowlevel/val.json"
    train_df = pd.read_json(train_path, lines=True)
    test_df = pd.read_json(val_path, lines=True)
    label_encoder = LabelEncoder()
    train_df["label"] = label_encoder.fit_transform(train_df["labels"].apply(lambda x: x[0]))
    test_df["label"] = label_encoder.transform(test_df["labels"].apply(lambda x: x[0]))

    # Применяем предобработку текстов (опционально)
    # train_df["text"] = train_df["text"].apply(basic_preprocessing)
    # test_df["text"] = test_df["text"].apply(basic_preprocessing)

    # tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")
    tokenizer = BertTokenizer.from_pretrained('ProsusAI/finbert')

    train_dataset = FinancialDataset(
        texts=train_df["text"],
        labels=train_df["label"],
        tokenizer=tokenizer,
        max_len=128
    )
    test_dataset = FinancialDataset(
        texts=test_df["text"],
        labels=test_df["label"],
        tokenizer=tokenizer,
        max_len=128
    )

    # model = XLMRobertaForSequenceClassification.from_pretrained(
    #     "xlm-roberta-base",
    #     num_labels=len(label_encoder.classes_)
    # )

    model = BertForSequenceClassification.from_pretrained(
        'ProsusAI/finbert',
        num_labels=len(label_encoder.classes_),
        ignore_mismatched_sizes=True
    )

    training_args = TrainingArguments(
        output_dir="outputs",
        num_train_epochs=15,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        learning_rate=3e-4,
        evaluation_strategy="epoch",
        save_strategy="no",
        logging_dir="logs",
        # load_best_model_at_end=True,
        metric_for_best_model="f1_micro",
        # greater_is_better=True,
        warmup_steps=100,
        weight_decay=2,
        report_to=[]
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )

    trainer.train()

    results = trainer.evaluate()
    print("Validation results:", results)

