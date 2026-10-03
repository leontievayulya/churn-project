"""Модуль для загрузки данных Telco Customer Churn."""

from pathlib import Path

import pandas as pd

# Путь к датасету относительно корня проекта
DEFAULT_DATA_PATH = Path("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")


def load_raw_data(path: Path | str = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Загружает исходный CSV с данными.

    Parameters
    ----------
    path : Path | str
        Путь к CSV-файлу.

    Returns
    -------
    pd.DataFrame
        Исходный датафрейм без обработки.

    Raises
    ------
    FileNotFoundError
        Если файл по указанному пути не найден.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {path}")

    return pd.read_csv(path)
