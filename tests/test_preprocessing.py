"""Тесты для модуля preprocessing."""

import pytest

from src.churn_project.data_loader import load_raw_data
from src.churn_project.preprocessing import clean_data, encode_features, preprocess


@pytest.fixture
def raw_df():
    """Исходный датафрейм для тестов."""
    return load_raw_data()


def test_clean_data_removes_customer_id(raw_df):
    """customerID удаляется."""
    df = clean_data(raw_df)
    assert "customerID" not in df.columns


def test_clean_data_total_charges_float(raw_df):
    """TotalCharges становится float, пропусков нет."""
    df = clean_data(raw_df)
    assert df["TotalCharges"].dtype == "float64"
    assert df["TotalCharges"].isnull().sum() == 0


def test_clean_data_churn_binary(raw_df):
    """Churn закодирован 0/1."""
    df = clean_data(raw_df)
    assert set(df["Churn"].unique()) == {0, 1}


def test_encode_features_no_object_columns(raw_df):
    """После кодирования нет колонок типа object/str."""
    df = encode_features(clean_data(raw_df))
    object_cols = df.select_dtypes(include=["object", "str"]).columns
    assert len(object_cols) == 0


def test_preprocess_shape(raw_df):
    """Финальная форма датасета (7043, 31)."""
    df = preprocess(raw_df)
    assert df.shape == (7043, 31)


def test_preprocess_no_nulls(raw_df):
    """После обработки пропусков нет."""
    df = preprocess(raw_df)
    assert df.isnull().sum().sum() == 0
