"""Модуль для обучения модели Telco Customer Churn."""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Настройки по умолчанию
RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET_COL = "Churn"


def split_data(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Разделяет данные на обучающую и тестовую выборки со стратификацией.

    Parameters
    ----------
    df : pd.DataFrame
        Обработанный датафрейм (все признаки числовые).
    test_size : float
        Доля тестовой выборки (по умолчанию 0.2).
    random_state : int
        Seed для воспроизводимости.

    Returns
    -------
    tuple
        X_train, X_test, y_train, y_test
    """
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test


def build_model(random_state: int = RANDOM_STATE) -> Pipeline:
    """Создаёт Pipeline с Random Forest.

    Параметры подобраны для задачи удержания:
    - class_weight='balanced' — компенсация дисбаланса классов.
    - max_depth=10 — защита от переобучения.
    - min_samples_leaf=5 — устойчивость к шуму.

    Parameters
    ----------
    random_state : int
        Seed для воспроизводимости.

    Returns
    -------
    Pipeline
        Готовый к обучению Pipeline.
    """
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=10,
                    min_samples_leaf=5,
                    class_weight="balanced",
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def train_model(
    model: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Pipeline:
    """Обучает модель на тренировочных данных.

    Parameters
    ----------
    model : Pipeline
        Необученный Pipeline.
    X_train : pd.DataFrame
        Признаки тренировочной выборки.
    y_train : pd.Series
        Целевая переменная тренировочной выборки.

    Returns
    -------
    Pipeline
        Обученная модель.
    """
    model.fit(X_train, y_train)
    return model
