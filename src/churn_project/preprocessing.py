"""Модуль для предобработки данных Telco Customer Churn.

Содержит функции:
- clean_data: очистка (TotalCharges, customerID, target)
- encode_features: кодирование категориальных признаков
- preprocess: полный пайплайн предобработки
"""

import pandas as pd

# Бинарные колонки и их маппинг в 0/1
BINARY_COLS_MAP = {
    "gender": {"Female": 0, "Male": 1},
    "Partner": {"No": 0, "Yes": 1},
    "Dependents": {"No": 0, "Yes": 1},
    "PhoneService": {"No": 0, "Yes": 1},
    "PaperlessBilling": {"No": 0, "Yes": 1},
}


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Очищает данные: обрабатывает TotalCharges, удаляет customerID, кодирует target.

    Parameters
    ----------
    df : pd.DataFrame
        Исходный датафрейм.

    Returns
    -------
    pd.DataFrame
        Очищенный датафрейм.
    """
    df = df.copy()

    # 1. TotalCharges: пробелы -> NaN, заполняем нулями, тип float
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    # 2. Удаляем customerID (уникальный идентификатор, не признак)
    df = df.drop(columns=["customerID"])

    # 3. Churn: Yes/No -> 1/0
    df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

    return df


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """Кодирует категориальные признаки: бинарные -> 0/1, многоклассовые -> One-Hot.

    Parameters
    ----------
    df : pd.DataFrame
        Датафрейм после clean_data.

    Returns
    -------
    pd.DataFrame
        Датафрейм со всеми числовыми колонками.
    """
    df = df.copy()

    # 1. Бинарные колонки -> 0/1
    for col, mapping in BINARY_COLS_MAP.items():
        df[col] = df[col].map(mapping)

    # 2. Оставшиеся строковые колонки -> One-Hot
    cat_cols = df.select_dtypes(include=["str", "object"]).columns.tolist()
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=int)

    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Полный пайплайн предобработки.

    Parameters
    ----------
    df : pd.DataFrame
        Исходный датафрейм.

    Returns
    -------
    pd.DataFrame
        Готовый к обучению датафрейм.
    """
    df = clean_data(df)
    df = encode_features(df)
    return df
