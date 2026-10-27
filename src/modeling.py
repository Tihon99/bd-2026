"""Общее для моделей: разбиение по соревнованиям, метрики, базовые модели."""
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def split_by_competition(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Train/test так, чтобы команды одного соревнования не попадали в обе части."""
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(df, groups=df["CompetitionId"]))
    return df.iloc[train_idx], df.iloc[test_idx]


def evaluate(y_true, y_score) -> dict:
    """ROC-AUC и PR-AUC (классы несбалансированы: в топ-10% — примерно десятая часть команд)."""
    return {
        "roc_auc": roc_auc_score(y_true, y_score),
        "pr_auc": average_precision_score(y_true, y_score),
    }


def baseline_models() -> dict:
    return {
        "constant": DummyClassifier(strategy="prior"),
        "logreg": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced"),
        ),
    }
