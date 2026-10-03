# Telco Customer Churn

ML-проект для предсказания оттока клиентов телеком-компании.

## О проекте

**Задача:** бинарная классификация — предсказать, уйдёт ли клиент.

**Датасет:** [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) — 7043 записи, 21 признак, целевая переменная `Churn` (Yes/No).

**Целевая метрика:** ROC-AUC.

## Результаты

**Финальная модель:** Random Forest с `class_weight='balanced'`.

| Метрика | Train | Test |
|---|---|---|
| Accuracy | 0.8213 | 0.7700 |
| Precision | 0.6178 | 0.5472 |
| Recall | 0.8562 | 0.7754 |
| F1-score | 0.7177 | 0.6416 |
| **ROC-AUC** | 0.9128 | **0.8441** |

**Почему Random Forest:**
- Лучший F1-score (0.6324) среди протестированных моделей.
- Высокий Recall (0.7888) — модель ловит ~79% реальных уходов.
- Умеренный разрыв Train ↔ Test (0.07) — нет сильного переобучения.

**Что было протестировано:** Logistic Regression (обычная и с балансировкой), Random Forest, Gradient Boosting с балансировкой.

## Структура проекта
churn_project/
├── data/
│ ├── raw/ # Исходный CSV (не коммитится)
│ └── processed/ # Обработанные данные (не коммитится)
├── notebooks/ # Jupyter-ноутбуки
├── scripts/
│ ├── explore_data.py # Разведочный анализ данных (черновик)
│ └── pipeline.py # End-to-end пайплайн
├── src/
│ └── churn_project/ # Production-код
│ ├── data_loader.py # Загрузка данных
│ ├── preprocessing.py # Очистка, кодирование
│ ├── train.py # Train/test split, модель
│ └── evaluate.py # Метрики и графики
├── tests/ # Тесты (pytest)
├── pyproject.toml # Зависимости (Poetry)
├── .pre-commit-config.yaml # Хуки pre-commit
├── ruff.toml # Настройки линтера
├── mypy.ini # Настройки проверки типов
└── README.md

## Стек технологий

- **Python 3.11+**
- **Poetry** — управление зависимостями
- **pandas, numpy** — обработка данных
- **scikit-learn** — ML-модели
- **matplotlib, seaborn** — визуализация
- **ruff, mypy** — линтеры и проверка типов
- **pre-commit** — автоматические проверки перед коммитом
- **pytest** — тесты

## Установка

```bash
# 1. Клонировать репозиторий
git clone <URL>
cd churn_project

# 2. Установить Poetry
pip install poetry
poetry config virtualenvs.in-project true

# 3. Установить зависимости
poetry install

# 4. Установить pre-commit хуки
poetry run pre-commit install
