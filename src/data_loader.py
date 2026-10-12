"""Загрузка таблиц Meta Kaggle из data/raw."""
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

# Читаем только нужные колонки: некоторые таблицы весят гигабайты
COLUMNS = {
    "Competitions": [
        "Id", "Slug", "Title", "HostSegmentTitle", "EnabledDate", "DeadlineDate",
        "RewardType", "RewardQuantity", "TotalTeams", "TotalCompetitors", "TotalSubmissions",
    ],
    "Users": ["Id", "UserName", "RegisterDate", "PerformanceTier"],
    "Teams": [
        "Id", "CompetitionId", "TeamLeaderId", "ScoreFirstSubmittedDate",
        "LastSubmissionDate", "Medal", "PrivateLeaderboardRank",
    ],
    "TeamMemberships": ["TeamId", "UserId", "RequestDate"],
    "Kernels": [
        "Id", "AuthorUserId", "CreationDate", "MadePublicDate",
        "TotalViews", "TotalComments", "TotalVotes",
    ],
    "Tags": ["Id", "Name"],
    "CompetitionTags": ["CompetitionId", "TagId"],
    "KernelTags": ["KernelId", "TagId"],
}

DATE_COLUMNS = {
    "Competitions": ["EnabledDate", "DeadlineDate"],
    "Users": ["RegisterDate"],
    "Teams": ["ScoreFirstSubmittedDate", "LastSubmissionDate"],
    "TeamMemberships": ["RequestDate"],
    "Kernels": ["CreationDate", "MadePublicDate"],
}

TIERS = {0: "Novice", 1: "Contributor", 2: "Expert", 3: "Master", 4: "Grandmaster", 5: "Staff"}


def load(name: str, nrows: int | None = None) -> pd.DataFrame:
    """Читает data/raw/<name>.csv, оставляет нужные колонки и парсит даты."""
    path = DATA_DIR / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} не найден — скачайте данные в data/raw/")

    df = pd.read_csv(path, usecols=COLUMNS.get(name), nrows=nrows, low_memory=False)
    for col in DATE_COLUMNS.get(name, []):
        df[col] = pd.to_datetime(df[col], format="mixed", errors="coerce")
    return df
