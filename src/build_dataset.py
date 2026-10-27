"""Датасет для модели: команды Featured-соревнований, признаки и цель (попадание в топ-10%).

Сборка долгая (Submissions читается по частям), поэтому результат кешируется в data/processed/.
Пересобрать из корня репозитория: python -m src.build_dataset
"""
from pathlib import Path

import pandas as pd

from src.data_loader import load
from src.features import team_features, submission_counts

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
DATASET_PATH = PROCESSED_DIR / "teams_dataset.parquet"

# Только то, что известно в начале соревнования (общее число посылок — нет, это утечка)
FEATURES = ["team_size", "max_prev_comps", "mean_prev_comps", "max_tier", "n_subs_first7d"]
TARGET = "top10"


def build() -> pd.DataFrame:
    comp = load("Competitions")
    teams = load("Teams")
    members = load("TeamMemberships")
    users = load("Users")

    featured = comp.loc[comp["HostSegmentTitle"] == "Featured", ["Id", "TotalTeams", "EnabledDate"]]
    df = teams.merge(featured, left_on="CompetitionId", right_on="Id", suffixes=("", "_comp"))
    df = df.dropna(subset=["PrivateLeaderboardRank"]).set_index("Id")

    df[TARGET] = df["PrivateLeaderboardRank"] / df["TotalTeams"] <= 0.1
    df["year"] = df["EnabledDate"].dt.year

    df = df.join(team_features(teams, members, users)).join(submission_counts(teams, comp))
    df["team_size"] = df["team_size"].fillna(1)
    df["n_subs_first7d"] = df["n_subs_first7d"].fillna(0)
    return df[["CompetitionId", "year"] + FEATURES + [TARGET]]


def get_dataset(rebuild: bool = False) -> pd.DataFrame:
    """Датасет из кеша; если кеша нет (или rebuild=True) — собирает заново."""
    if DATASET_PATH.exists() and not rebuild:
        return pd.read_parquet(DATASET_PATH)
    df = build()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(DATASET_PATH)
    return df


if __name__ == "__main__":
    data = get_dataset(rebuild=True)
    print(f"{len(data)} команд, доля top10: {data[TARGET].mean():.3f}")
    print(f"сохранено: {DATASET_PATH}")
