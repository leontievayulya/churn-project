"""Модуль для оценки качества модели Telco Customer Churn."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline


def compute_metrics(
    model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
) -> dict[str, float]:
    """Считает основные метрики классификации.

    Parameters
    ----------
    model : Pipeline
        Обученная модель с методами predict и predict_proba.
    X : pd.DataFrame
        Признаки.
    y : pd.Series
        Истинные метки.

    Returns
    -------
    dict
        Метрики: accuracy, precision, recall, f1, roc_auc.
    """
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    return {
        "accuracy": float(accuracy_score(y, y_pred)),
        "precision": float(precision_score(y, y_pred)),
        "recall": float(recall_score(y, y_pred)),
        "f1": float(f1_score(y, y_pred)),
        "roc_auc": float(roc_auc_score(y, y_proba)),
    }


def print_metrics(metrics: dict[str, float], name: str) -> None:
    """Печатает метрики в читаемом виде.

    Parameters
    ----------
    metrics : dict
        Словарь с метриками.
    name : str
        Название выборки (Train/Test).
    """
    print(f"\n{'=' * 50}")
    print(f"МЕТРИКИ: {name}")
    print(f"{'=' * 50}")
    for metric_name, value in metrics.items():
        print(f"  {metric_name:<12} {value:.4f}")


def plot_confusion_matrix(
    model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    save_path: Path | None = None,
) -> None:
    """Строит матрицу ошибок.

    Parameters
    ----------
    model : Pipeline
        Обученная модель.
    X : pd.DataFrame
        Признаки.
    y : pd.Series
        Истинные метки.
    save_path : Path | None
        Путь для сохранения (если None — только показ).
    """
    y_pred = model.predict(X)
    cm = confusion_matrix(y, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Остался (0)", "Ушёл (1)"],
        yticklabels=["Остался (0)", "Ушёл (1)"],
        cbar=False,
        annot_kws={"size": 14},
    )
    plt.title("Матрица ошибок")
    plt.ylabel("Истина")
    plt.xlabel("Прогноз")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.show()


def plot_roc_curve(
    model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    save_path: Path | None = None,
) -> None:
    """Строит ROC-кривую.

    Parameters
    ----------
    model : Pipeline
        Обученная модель.
    X : pd.DataFrame
        Признаки.
    y : pd.Series
        Истинные метки.
    save_path : Path | None
        Путь для сохранения (если None — только показ).
    """
    y_proba = model.predict_proba(X)[:, 1]
    fpr, tpr, _ = roc_curve(y, y_proba)
    auc = roc_auc_score(y, y_proba)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color="#fb8500", linewidth=2, label=f"ROC-кривая (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Случайное угадывание (AUC = 0.500)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC-кривая")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.show()


def print_classification_report(
    model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
) -> None:
    """Печатает classification_report.

    Parameters
    ----------
    model : Pipeline
        Обученная модель.
    X : pd.DataFrame
        Признаки.
    y : pd.Series
        Истинные метки.
    """
    y_pred = model.predict(X)
    print("\nClassification Report:")
    print(
        classification_report(
            y,
            y_pred,
            target_names=["Остался (0)", "Ушёл (1)"],
        )
    )
