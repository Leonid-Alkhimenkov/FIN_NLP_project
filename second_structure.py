import os
import re
import numpy as np
import pandas as pd
import torch
import regex as re
import gc

from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from transformers import (
    M2M100ForConditionalGeneration,
    M2M100Tokenizer,
    BertTokenizer,
    BertForSequenceClassification,
    XLMRobertaTokenizer,
    XLMRobertaForSequenceClassification,
    Trainer,
    TrainingArguments
)
from torch.utils.data import Dataset
from tqdm.auto import tqdm

os.environ["WANDB_DISABLED"] = "true"

LANGUAGE_NAME_TO_CODE = {
    "English": "en",
    "Turkish": "tr",
    "Danish": "da",
    "Spanish": "es",
    "Polish": "pl",
    "Greek": "el",
    "Finnish": "fi",
    "Hebrew": "he",
    "Japanese": "ja",
    "Hungarian": "hu",
    "Norwegian": "no",
    "Russian": "ru",
    "Italian": "it",
    "Icelandic": "is",
    "Swedish": "sv"
}

class M2M100TranslationManager:
    """
    Класс для управления моделью-переводчиком M2M-100.
    Переводит тексты с заданных языков на английский.
    """

    def __init__(self, target_lang="en", device='cpu'):
        """
        Инициализирует модель-переводчик.

        :param target_lang: Целевой язык перевода (по умолчанию 'en').
        :param device: Устройство ('cpu' или 'cuda').
        """
        self.target_lang = target_lang
        self.tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
        self.model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M").to(device)
        self.device = device
        self.tokenizer.tgt_lang = self.target_lang
        self.cache = {}

        print(f"Модель-переводчик загружена на устройство: {next(self.model.parameters()).device}")

    def translate_batch(self, texts, src_lang_codes, batch_size=8):
        """
        Переводит список текстов с исходных языков на целевой язык.

        :param texts: Список строк на исходном языке.
        :param src_lang_codes: Список двухбуквенных кодов исходных языков.
        :param batch_size: Размер батча для перевода.
        :return: Список переведённых строк на целевом языке.
        """
        translated_texts = [''] * len(texts)
        lang_to_texts = {}
        indices_per_lang = {}

        for idx, (text, lang_code) in enumerate(zip(texts, src_lang_codes)):
            if lang_code == self.target_lang:
                translated_texts[idx] = text
                continue
            key = (text, lang_code)
            if key in self.cache:
                translated_texts[idx] = self.cache[key]
            else:
                if lang_code not in lang_to_texts:
                    lang_to_texts[lang_code] = []
                    indices_per_lang[lang_code] = []
                lang_to_texts[lang_code].append(text)
                indices_per_lang[lang_code].append(idx)

        for lang_code, texts_lang in lang_to_texts.items():
            indices_lang = indices_per_lang[lang_code]
            if not texts_lang:
                continue
            try:
                self.tokenizer.src_lang = lang_code
                for j in tqdm(range(0, len(texts_lang), batch_size), desc=f"Translating {lang_code}"):
                    batch_texts = texts_lang[j:j+batch_size]
                    batch_indices = indices_lang[j:j+batch_size]

                    encoded = self.tokenizer(batch_texts, return_tensors="pt", truncation=True, padding=True).to(self.device)

                    try:
                        generated_tokens = self.model.generate(
                            **encoded,
                            forced_bos_token_id=self.tokenizer.get_lang_id(self.target_lang),
                            max_length=512
                        )
                    except Exception as e:
                        print(f"Ошибка при генерации перевода: {e}")
                        for idx_in_batch in batch_indices:
                            translated_texts[idx_in_batch] = ""

                    translated_batch = [self.tokenizer.decode(t, skip_special_tokens=True) for t in generated_tokens]

                    for idx_in_batch, translated_text in zip(batch_indices, translated_batch):
                        translated_texts[idx_in_batch] = translated_text
                        self.cache[(batch_texts[0], lang_code)] = translated_text

            except Exception as e:
                print(f"Ошибка при переводе с языка '{lang_code}': {e}")
                for idx_in_batch in indices_lang:
                    translated_texts[idx_in_batch] = ""  # Или texts[idx_in_batch]

            if self.device == 'cuda':
                torch.cuda.empty_cache()
                gc.collect()

        return translated_texts

class FinancialDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        """
        :param texts: pd.Series с переведёнными текстами на английский.
        :param labels: pd.Series с метками.
        :param tokenizer: Токенизатор для модели классификации.
        :param max_len: Максимальная длина токенов.
        """
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
            "input_ids": encoding["input_ids"].squeeze(0),  # torch.Size([max_len])
            "attention_mask": encoding["attention_mask"].squeeze(0),  # torch.Size([max_len])
            "labels": torch.tensor(label, dtype=torch.long)
        }


def basic_preprocessing(text: str) -> str:
    """
    Базовая функция предобработки для многоязычного текста:
    1) Приведение к нижнему регистру
    2) Удаление всего, что не является буквой, цифрой, пунктуацией или пробельным символом (в Unicode)
    3) Удаление лишних пробелов
    Примечание: \p{L} захватывает все буквы, в т.ч. японские (漢字, カタカナ, ひらがな),
    \p{N} — любые цифры, \p{P} — пунктуацию, \p{Z} — любые пробельные символы (включая неразрывные).
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
    # Путь к данным (train.json, val.json)
    train_path = Path(__file__).parent / "0-all-languages-lowlevel/train.json"
    val_path = Path(__file__).parent / "0-all-languages-lowlevel/val.json"

    try:
        train_df = pd.read_json(train_path, lines=True)
        test_df = pd.read_json(val_path, lines=True)
    except Exception as e:
        print(f"Ошибка при загрузке данных: {e}")
        exit(1)

    train_df['lang_code'] = train_df['lang'].map(LANGUAGE_NAME_TO_CODE)
    test_df['lang_code'] = test_df['lang'].map(LANGUAGE_NAME_TO_CODE)

    unknown_train = train_df['lang_code'].isnull()
    unknown_test = test_df['lang_code'].isnull()

    if unknown_train.any() or unknown_test.any():
        print("Найдены неизвестные языковые коды. Пожалуйста, проверьте данные.")
        if unknown_train.any():
            print("Неизвестные в тренировочном наборе:")
            print(train_df[unknown_train][['id', 'lang']])
        if unknown_test.any():
            print("Неизвестные в валидационном наборе:")
            print(test_df[unknown_test][['id', 'lang']])
        train_df = train_df[~unknown_train]
        test_df = test_df[~unknown_test]

    label_encoder = LabelEncoder()
    try:
        train_df["label"] = label_encoder.fit_transform(train_df["labels"].apply(lambda x: x[0]))
        test_df["label"] = label_encoder.transform(test_df["labels"].apply(lambda x: x[0]))
    except Exception as e:
        print(f"Ошибка при кодировании меток: {e}")
        exit(1)

    # Применяем предобработку текстов (опционально)
    # train_df["text"] = train_df["text"].apply(basic_preprocessing)
    # test_df["text"] = test_df["text"].apply(basic_preprocessing)

    device = 'cpu'  # Перевод выполняется на CPU для экономии памяти GPU
    translator = M2M100TranslationManager(target_lang="en", device=device)

    def preliminary_translation(translator, df, dataset_name="train"):
        print(f"Начинаем предварительный перевод {dataset_name} набора данных...")
        translated_texts = translator.translate_batch(
            texts=df["text"].tolist(),
            src_lang_codes=df["lang_code"].tolist(),
            batch_size=8
        )
        df["translated_text"] = translated_texts
        print(f"Перевод {dataset_name} набора данных завершён.")
        torch.cuda.empty_cache()
        gc.collect()

    preliminary_translation(translator, train_df, dataset_name="тренировочного")
    preliminary_translation(translator, test_df, dataset_name="валидационного")

    torch.cuda.empty_cache()
    gc.collect()

    try:
        tokenizer = BertTokenizer.from_pretrained('ProsusAI/finbert')
        # tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")
    except Exception as e:
        print(f"Ошибка при загрузке токенизатора: {e}")
        exit(1)

    train_dataset = FinancialDataset(
        texts=train_df["translated_text"],
        labels=train_df["label"],
        tokenizer=tokenizer,
        max_len=128
    )
    test_dataset = FinancialDataset(
        texts=test_df["translated_text"],
        labels=test_df["label"],
        tokenizer=tokenizer,
        max_len=128
    )

    print("Загружаем модель...")
    try:
        model = BertForSequenceClassification.from_pretrained(
            'ProsusAI/finbert',
            num_labels=len(label_encoder.classes_),
            ignore_mismatched_sizes=True  # Для корректной загрузки новых классификаторов
        )
        # model = XLMRobertaForSequenceClassification.from_pretrained(
        #     "xlm-roberta-base",
        #     num_labels=len(label_encoder.classes_)
        # )
    except Exception as e:
        print(f"Ошибка при загрузке модели FinBERT: {e}")
        exit(1)

    training_args = TrainingArguments(
        output_dir="outputs",
        num_train_epochs=7,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        learning_rate=2.5e-5,
        evaluation_strategy="epoch",
        save_strategy="no",
        logging_dir="logs",
        metric_for_best_model="f1_micro",
        warmup_steps=100,
        weight_decay=0.03,
        report_to=[]
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )

    # Обучаем модель
    print("Начинаем обучение модели...")
    try:
        trainer.train()
    except Exception as e:
        print(f"Ошибка при обучении модели: {e}")
        exit(1)

    print("Проводим оценку модели на валидационном наборе...")
    try:
        results = trainer.evaluate()
        print("Validation results:", results)
    except Exception as e:
        print(f"Ошибка при оценке модели: {e}")
        exit(1)