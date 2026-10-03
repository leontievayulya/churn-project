# Telco Customer Churn

ML-проект для предсказания оттока клиентов телеком-компании.

## О проекте

**Задача:** бинарная классификация — предсказать, уйдёт ли клиент из телеком-компании.

**Датасет:** [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
7043 записи, 21 признак, целевая переменная `Churn` (Yes/No).

**Целевая метрика:** ROC-AUC.

## Структура проекта

churn_project/
├── data/
│ ├── raw/
│ └── processed/
├── notebooks/
├── src/
│ └── churn_project/ # Production-код
│ └── init.py
├── tests/ # Тесты
├── .gitignore
├── .pre-commit-config.yaml
├── pyproject.toml
└── README.md


## 🛠 Стек технологий

- Python 3.10+
- Poetry
- pandas, numpy, scikit-learn
- ruff, mypy, pre-commit

## Установка

```bash
poetry install
poetry shell
