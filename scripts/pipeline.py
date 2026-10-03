"""Полный ML-пайплайн для Telco Customer Churn.

Тонкая обёртка над модулями src/churn_project.

Запуск:
    python scripts/pipeline.py
"""

import sys
from pathlib import Path

# Добавляем корень проекта в путь (чтобы src был доступен)
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.churn_project.data_loader import load_raw_data
from src.churn_project.evaluate import compute_metrics, print_metrics
from src.churn_project.preprocessing import preprocess
from src.churn_project.train import build_model, split_data, train_model


def main() -> None:
    """Запускает весь пайплайн: загрузка → обработка → обучение → оценка."""
    print("Загрузка данных...")
    df = load_raw_data()
    print(f"  Загружено: {df.shape}")

    print("Препроцессинг...")
    df = preprocess(df)
    print(f"  После обработки: {df.shape}")

    print("Train/test split...")
    X_train, X_test, y_train, y_test = split_data(df)
    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")

    print("Обучение Random Forest...")
    model = build_model()
    train_model(model, X_train, y_train)

    print("\nОценка на TEST:")
    test_metrics = compute_metrics(model, X_test, y_test)
    print_metrics(test_metrics, "Test")

    print("\nОценка на TRAIN:")
    train_metrics = compute_metrics(model, X_train, y_train)
    print_metrics(train_metrics, "Train")


if __name__ == "__main__":
    main()
