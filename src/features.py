"""Общие признаки команд: размер, опыт участников, посылки."""
import pandas as pd

from src.data_loader import iter_chunks


def team_sizes(members: pd.DataFrame) -> pd.Series:
    """Число участников в каждой команде (TeamId -> team_size)."""
    return members.groupby("TeamId").size().rename("team_size")


def member_experience(teams: pd.DataFrame, members: pd.DataFrame) -> pd.DataFrame:
    """Для каждого участника команды — в скольких соревнованиях он был до этого.

    Порядок соревнований определяем по дате первой посылки команды.
    """
    part = members[["TeamId", "UserId"]].merge(
        teams[["Id", "CompetitionId", "ScoreFirstSubmittedDate"]],
        left_on="TeamId", right_on="Id",
    )
    part = part.dropna(subset=["ScoreFirstSubmittedDate"])
    part = part.sort_values(["UserId", "ScoreFirstSubmittedDate"])
    part["prev_comps"] = part.groupby("UserId").cumcount()
    return part[["TeamId", "UserId", "CompetitionId", "prev_comps"]]


def team_features(teams: pd.DataFrame, members: pd.DataFrame, users: pd.DataFrame) -> pd.DataFrame:
    """Признаки команды: размер, опыт (макс./средний), максимальный уровень участников.

    Внимание: PerformanceTier — текущий уровень, а не на момент соревнования.
    """
    exp = member_experience(teams, members).merge(
        users[["Id", "PerformanceTier"]], left_on="UserId", right_on="Id", how="left"
    )
    feats = exp.groupby("TeamId").agg(
        max_prev_comps=("prev_comps", "max"),
        mean_prev_comps=("prev_comps", "mean"),
        max_tier=("PerformanceTier", "max"),
    )
    return feats.join(team_sizes(members), how="outer")


def submission_counts(teams: pd.DataFrame, competitions: pd.DataFrame, first_days: int = 7) -> pd.DataFrame:
    """Число посылок команды: всего и за первые first_days дней соревнования.

    Submissions очень большая, поэтому читаем её по частям.
    """
    start = teams[["Id", "CompetitionId"]].merge(
        competitions[["Id", "EnabledDate"]], left_on="CompetitionId", right_on="Id", suffixes=("", "_c")
    ).set_index("Id")["EnabledDate"]

    parts = []
    for chunk in iter_chunks("Submissions"):
        # в CSV флаг может прийти как bool или как строка "True"/"False" — приводим к строке
        after_deadline = chunk["IsAfterDeadline"].astype(str).str.lower().eq("true")
        chunk = chunk[~after_deadline]
        days = (chunk["SubmissionDate"] - chunk["TeamId"].map(start)).dt.days
        chunk = chunk.assign(early=(days >= 0) & (days < first_days))
        parts.append(chunk.groupby("TeamId").agg(n_subs=("early", "size"), n_subs_early=("early", "sum")))

    counts = pd.concat(parts).groupby(level=0).sum()
    return counts.rename(columns={"n_subs_early": f"n_subs_first{first_days}d"})
