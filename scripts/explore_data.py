# %%  Импорты
import numpy as np
import pandas as pd

print(f"pandas version: {pd.__version__}")
print(f"numpy version: {np.__version__}")

# %%
# %%  Загрузка датасета
DATA_PATH = r"D:\Yulya_study\project\churn_project\data\raw\WA_Fn-UseC_-Telco-Customer-Churn.csv"

df = pd.read_csv(DATA_PATH)

print(f"Размер датасета: {df.shape}")
print(f"Строк: {df.shape[0]}, Колонок: {df.shape[1]}")

# %%
print(df.head())
df.info()
df.describe()
# %%
print(df["Churn"].value_counts())
print(f"\nДоля оттока: {df['Churn'].value_counts(normalize=True)}")
cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
print(f"\nКатегориальные ({len(cat_cols)}):")
for col in cat_cols:
    n_unique = df[col].nunique()
    print(f"  - {col} ({n_unique} уник. значений)")


# %%
num_cols = df.select_dtypes(include=["float64", "int64"]).columns.to_list()
print(f"всего числовых колонок-{len(num_cols)}")
print(f"Уникальных значений: {df['TotalCharges'].nunique()}")
blank_count = (df["TotalCharges"].astype(str).str.strip() == "").sum()
print(f"\nПустых (пробелов) в TotalCharges: {blank_count}")
df[num_cols].describe()
# %%
# %%  Кто эти 11 строк с пропусками в TotalCharges?
mask = df["TotalCharges"].astype(str).str.strip() == ""

print("=" * 60)
print("СТРОКИ С ПРОПУСКАМИ В TotalCharges")
print("=" * 60)
print(f"Количество: {mask.sum()}")

# Смотрим, что у них с tenure и MonthlyCharges
print("\nИх tenure (сколько месяцев с нами):")
print(df.loc[mask, "tenure"].value_counts().sort_index())

print("\nИх MonthlyCharges (ежемесячный платёж):")
print(df.loc[mask, "MonthlyCharges"].describe())

# Гипотеза: это новые клиенты с tenure=0?
print("\nИх Churn (ушли или остались):")
print(df.loc[mask, "Churn"].value_counts())
# %%
df["TotalCharges"] = df["TotalCharges"].fillna(0)
print(df.isnull().sum().sum(), "пропусков")
# %%
df = df.drop(columns=["customerID"])
df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})
df["gender"] = df["gender"].map({"Female": 0, "Male": 1})
df["Partner"] = df["Partner"].map({"No": 0, "Yes": 1})
df["Dependents"] = df["Dependents"].map({"No": 0, "Yes": 1})
df["PhoneService"] = df["PhoneService"].map({"No": 0, "Yes": 1})
df["PaperlessBilling"] = df["PaperlessBilling"].map({"No": 0, "Yes": 1})
print(f"Форма датасета: {df.shape}")
print(f"Всего пропусков: {df.isnull().sum().sum()}")
print(f"Колонок типа object/str: {df.select_dtypes(include=['object', 'str']).shape[1]}")
df.info()
# %%
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(0)
print("ИТОГОВЫЕ ТИПЫ:")
print("=" * 50)
print(df.dtypes.value_counts())
# %%
# %%  One-Hot Encoding для многоклассовых колонок
cat_cols = df.select_dtypes(include=["str", "object"]).columns.tolist()
for col in cat_cols:
    n = df[col].nunique()
    print(f"  - {col} ({n} значений)")

# One-Hot Encoding
df = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=int)

print(f"Пропусков: {df.isnull().sum().sum()}")
print("\nТипы данных:")
print(df.dtypes.value_counts())
# %%
from sklearn.model_selection import train_test_split

X = df.drop(columns=["Churn"])
y = df["Churn"]
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,  # 20% на тест
    random_state=42,  # воспроизводимость
    stratify=y,  # сохранить пропорции классов
)
print("\nДоля класса 1 (ушёл):")
print(f"  Train: {y_train.mean():.4f}")
print(f"  Test:  {y_test.mean():.4f}")
print(f"  Всего: {y.mean():.4f}")
# %%
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

model = Pipeline(
    [("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000, random_state=42))]
)

model.fit(X_train, y_train)
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)
y_train_proba = model.predict_proba(X_train)[:, 1]
y_test_proba = model.predict_proba(X_test)[:, 1]


def print_metrics(y_true, y_pred, y_proba, name):
    print(f"\n{'=' * 60}")
    print(f"МЕТРИКИ: {name}")
    print(f"{'=' * 60}")
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_true, y_pred):.4f}")
    print(f"F1-score:  {f1_score(y_true, y_pred):.4f}")
    print(f"ROC-AUC:   {roc_auc_score(y_true, y_proba):.4f}")


print_metrics(y_train, y_train_pred, y_train_proba, "TRAIN")
print_metrics(y_test, y_test_pred, y_test_proba, "TEST")
# %%
# %%  Матрица ошибок и полный отчёт
import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, y_true, y_pred, name in [
    (axes[0], y_train, y_train_pred, "Train"),
    (axes[1], y_test, y_test_pred, "Test"),
]:
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax,
        xticklabels=["Остался (0)", "Ушёл (1)"],
        yticklabels=["Остался (0)", "Ушёл (1)"],
        cbar=False,
        annot_kws={"size": 14},
    )
    ax.set_title(f"Матрица ошибок — {name}")
    ax.set_ylabel("Истина")
    ax.set_xlabel("Прогноз")

plt.tight_layout()
plt.show()

# Полный отчёт
print("\n" + "=" * 60)
print("CLASSIFICATION REPORT (TEST)")
print("=" * 60)
print(classification_report(y_test, y_test_pred, target_names=["Остался (0)", "Ушёл (1)"]))
# %%
# %%  Logistic Regression с class_weight='balanced'
from sklearn.linear_model import LogisticRegression

model_balanced = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")),
    ]
)

model_balanced.fit(X_train, y_train)

y_train_pred_b = model_balanced.predict(X_train)
y_test_pred_b = model_balanced.predict(X_test)
y_train_proba_b = model_balanced.predict_proba(X_train)[:, 1]
y_test_proba_b = model_balanced.predict_proba(X_test)[:, 1]

print("=" * 60)
print("СРАВНЕНИЕ: обычная LogReg vs class_weight='balanced'")
print("=" * 60)
print(f"{'Метрика':<15} {'Обычная':>12} {'Balanced':>12}")
print("-" * 40)

for name, metric_fn in [
    ("Accuracy", accuracy_score),
    ("Precision", precision_score),
    ("Recall", recall_score),
    ("F1", f1_score),
    ("ROC-AUC", roc_auc_score),
]:
    old = (
        metric_fn(y_test, y_test_pred) if name != "ROC-AUC" else roc_auc_score(y_test, y_test_proba)
    )
    new = (
        metric_fn(y_test, y_test_pred_b)
        if name != "ROC-AUC"
        else roc_auc_score(y_test, y_test_proba_b)
    )
    print(f"{name:<15} {old:>12.4f} {new:>12.4f}")

# Матрица ошибок новой модели
print("\n" + "=" * 60)
print("МАТРИЦА ОШИБОК (balanced) — TEST")
print("=" * 60)
cm_b = confusion_matrix(y_test, y_test_pred_b)
print(f"TN: {cm_b[0, 0]}, FP: {cm_b[0, 1]}")
print(f"FN: {cm_b[1, 0]}, TP: {cm_b[1, 1]}")
# %%
# %%  Random Forest
from sklearn.ensemble import RandomForestClassifier

# Random Forest с балансировкой классов
rf_model = Pipeline(
    [
        ("scaler", StandardScaler()),
        (
            "clf",
            RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_leaf=5,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ]
)

# Обучение
rf_model.fit(X_train, y_train)

# Предсказания
y_train_pred_rf = rf_model.predict(X_train)
y_test_pred_rf = rf_model.predict(X_test)
y_train_proba_rf = rf_model.predict_proba(X_train)[:, 1]
y_test_proba_rf = rf_model.predict_proba(X_test)[:, 1]


# Метрики
def print_metrics_short(y_true, y_pred, y_proba, name):
    print(f"\n{name}")
    print(f"  Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"  Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"  Recall:    {recall_score(y_true, y_pred):.4f}")
    print(f"  F1-score:  {f1_score(y_true, y_pred):.4f}")
    print(f"  ROC-AUC:   {roc_auc_score(y_true, y_proba):.4f}")


print("=" * 60)
print("RANDOM FOREST (n_estimators=200, max_depth=10)")
print("=" * 60)
print_metrics_short(y_train, y_train_pred_rf, y_train_proba_rf, "TRAIN")
print_metrics_short(y_test, y_test_pred_rf, y_test_proba_rf, "TEST")
# %%
# %%  Gradient Boosting
# %%  Gradient Boosting с балансировкой классов
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight

# Считаем веса для балансировки
sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)

# Тот же GB, но с весами
gb_balanced_model = Pipeline(
    [
        ("scaler", StandardScaler()),
        (
            "clf",
            GradientBoostingClassifier(
                n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=42
            ),
        ),
    ]
)

gb_balanced_model.fit(X_train, y_train, clf__sample_weight=sample_weights)

# Предсказания
y_train_pred_gbb = gb_balanced_model.predict(X_train)
y_test_pred_gbb = gb_balanced_model.predict(X_test)
y_train_proba_gbb = gb_balanced_model.predict_proba(X_train)[:, 1]
y_test_proba_gbb = gb_balanced_model.predict_proba(X_test)[:, 1]

print("=" * 60)
print("GRADIENT BOOSTING (с балансировкой классов)")
print("=" * 60)
print_metrics_short(y_train, y_train_pred_gbb, y_train_proba_gbb, "TRAIN")
print_metrics_short(y_test, y_test_pred_gbb, y_test_proba_gbb, "TEST")  # %%

# %%
# %%  Сводная таблица и визуализация всех моделей
import matplotlib.pyplot as plt
import pandas as pd

# Собираем метрики всех моделей
results = {
    "LogReg обычная": {
        "Accuracy": 0.8070,
        "Precision": 0.6584,
        "Recall": 0.5668,
        "F1": 0.6092,
        "ROC-AUC": 0.8418,
    },
    "LogReg balanced": {
        "Accuracy": 0.7402,
        "Precision": 0.5069,
        "Recall": 0.7861,
        "F1": 0.6164,
        "ROC-AUC": 0.8414,
    },
    "Random Forest": {
        "Accuracy": 0.7566,
        "Precision": 0.5277,
        "Recall": 0.7888,
        "F1": 0.6324,
        "ROC-AUC": 0.8433,
    },
    "GB balanced": {
        "Accuracy": 0.7410,
        "Precision": 0.5077,
        "Recall": 0.7888,
        "F1": 0.6178,
        "ROC-AUC": 0.8444,
    },
}

results_df = pd.DataFrame(results).T

print("=" * 80)
print("СВОДНАЯ ТАБЛИЦА ВСЕХ МОДЕЛЕЙ (TEST)")
print("=" * 80)
print(results_df.round(4).to_string())

# Визуализация
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Метрики, зависящие от порога
results_df[["Accuracy", "Precision", "Recall", "F1"]].plot(
    kind="bar", ax=axes[0], colormap="viridis"
)
axes[0].set_title("Метрики, зависящие от порога (Test)")
axes[0].set_ylabel("Значение")
axes[0].set_ylim(0.4, 0.9)
axes[0].legend(loc="lower right")
axes[0].grid(axis="y", alpha=0.3)
axes[0].tick_params(axis="x", rotation=30)

# ROC-AUC
results_df["ROC-AUC"].plot(kind="bar", ax=axes[1], color="#fb8500")
axes[1].axhline(y=0.5, color="red", linestyle="--", label="Random (0.5)")
axes[1].set_title("ROC-AUC (Test) — главная метрика")
axes[1].set_ylabel("ROC-AUC")
axes[1].set_ylim(0.8, 0.86)
axes[1].legend()
axes[1].grid(axis="y", alpha=0.3)
axes[1].tick_params(axis="x", rotation=30)

plt.tight_layout()
plt.show()
