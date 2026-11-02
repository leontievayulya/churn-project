import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Добавляем корень проекта в путь, чтобы импортировать src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.churn_project.data_loader import load_raw_data
from src.churn_project.evaluate import compute_metrics
from src.churn_project.preprocessing import preprocess
from src.churn_project.train import build_model, split_data

# Настройка стиля графиков
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["font.size"] = 11
sns.set_theme(style="whitegrid")


def main() -> None:
    print("1. Загрузка и препроцессинг данных...")
    df_raw = load_raw_data()
    df = preprocess(df_raw)
    X_train, X_test, y_train, y_test = split_data(df)
    print(f"   Train: {X_train.shape}, Test: {X_test.shape}")

    # ============================================================
    # 2. Разведочный анализ (EDA)
    # ============================================================
    print("\n2. Построение графиков EDA...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Распределение классов
    counts = y_train.value_counts().sort_index()
    axes[0].bar(["Остался (0)", "Ушёл (1)"], counts.values, color=["#219ebc", "#fb8500"])
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 50, f"{v}\n({v/len(y_train)*100:.1f}%)", ha="center", fontweight="bold")
    axes[0].set_title("Распределение целевой переменной (Train)")
    axes[0].set_ylabel("Количество клиентов")

    # Корреляционная матрица (топ-10 признаков)
    corr = X_train.corrwith(y_train).abs().sort_values(ascending=False)
    top_features = corr.head(10).index
    sns.heatmap(X_train[top_features].corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=axes[1])
    axes[1].set_title("Корреляция топ-10 признаков")

    plt.tight_layout()
    plt.show()

    # ============================================================
    # 3. Обучение и сравнение моделей
    # ============================================================
    print("\n3. Обучение моделей...")

    # Определяем модели (наша production-модель + альтернативы для сравнения)
    models = {
        "LogReg (обычная)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42))
        ]),
        "LogReg (balanced)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"))
        ]),
        "Random Forest (prod)": build_model(),  # Импортируем из src
        "Gradient Boosting (bal)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(random_state=42))
        ]),
    }

    # Автоматический сбор метрик
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        # Используем функцию из evaluate.py для автоматического расчета
        results[name] = compute_metrics(model, X_test, y_test)

    # Создаем DataFrame для красивого вывода
    results_df = pd.DataFrame(results).T.rename(columns={
        "accuracy": "Accuracy",
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1",
        "roc_auc": "ROC-AUC",
    })
    print("\n" + "=" * 80)
    print("СВОДНАЯ ТАБЛИЦА ВСЕХ МОДЕЛЕЙ (TEST)")
    print("=" * 80)
    print(results_df.round(4).to_string())

    # ============================================================
    # 4. Визуализация сравнения моделей
    # ============================================================
    print("\n4. Построение сравнительных графиков...")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Метрики, зависящие от порога
    results_df[["Accuracy", "Precision", "Recall", "F1"]].plot(
        kind="bar", ax=axes[0], colormap="viridis", edgecolor="black"
    )
    axes[0].set_title("Метрики (Test)")
    axes[0].set_ylabel("Значение")
    axes[0].set_ylim(0.4, 0.9)
    axes[0].legend(loc="lower right")
    axes[0].tick_params(axis="x", rotation=30)

    # ROC-AUC
    results_df["ROC-AUC"].plot(kind="bar", ax=axes[1], color="#fb8500", edgecolor="black")
    axes[1].axhline(y=0.5, color="red", linestyle="--", label="Random (0.5)")
    axes[1].set_title("ROC-AUC (Test)")
    axes[1].set_ylabel("ROC-AUC")
    axes[1].set_ylim(0.8, 0.86)
    axes[1].legend()
    axes[1].tick_params(axis="x", rotation=30)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
