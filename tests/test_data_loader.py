"""Тесты для модуля data_loader."""

import pandas as pd
import pytest

from src.churn_project.data_loader import load_raw_data


def test_load_raw_data_returns_dataframe():
    """Загрузка возвращает DataFrame с ожидаемой формой."""
    df = load_raw_data()
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (7043, 21)


def test_load_raw_data_has_expected_columns():
    """Датасет содержит ключевые колонки."""
    df = load_raw_data()
    expected_cols = ["customerID", "gender", "tenure", "MonthlyCharges", "TotalCharges", "Churn"]
    for col in expected_cols:
        assert col in df.columns, f"Колонка {col} отсутствует"


def test_load_raw_data_wrong_path_raises():
    """Неверный путь вызывает FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_raw_data("nonexistent/file.csv")
