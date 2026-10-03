"""Тесты для модуля train."""

import pytest

from src.churn_project.data_loader import load_raw_data
from src.churn_project.preprocessing import preprocess
from src.churn_project.train import build_model, split_data, train_model


@pytest.fixture
def processed_df():
    """Обработанный датафрейм."""
    return preprocess(load_raw_data())


def test_split_data_shapes(processed_df):
    """Train/test split даёт ожидаемые размеры."""
    X_train, X_test, y_train, y_test = split_data(processed_df)
    assert X_train.shape[0] == 5634
    assert X_test.shape[0] == 1409
    assert X_train.shape[1] == X_test.shape[1] == 30


def test_split_data_stratified(processed_df):
    """Доли класса 1 сохраняются в train и test."""
    X_train, X_test, y_train, y_test = split_data(processed_df)
    diff = abs(y_train.mean() - y_test.mean())
    assert diff < 0.01, "Стратификация нарушена"


def test_build_model_returns_pipeline(processed_df):
    """build_model возвращает Pipeline."""
    model = build_model()
    assert hasattr(model, "fit")
    assert hasattr(model, "predict")


def test_train_model_fits(processed_df):
    """Обучение не падает и модель делает предсказания."""
    X_train, X_test, y_train, y_test = split_data(processed_df)
    model = train_model(build_model(), X_train, y_train)
    preds = model.predict(X_test)
    assert len(preds) == len(y_test)
    assert set(preds).issubset({0, 1})
