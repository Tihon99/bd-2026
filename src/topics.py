"""Тематика соревнований по тегам: CV / NLP / табличные данные / временные ряды."""
import pandas as pd

# Ключевые слова в названии тега -> тема. Порядок = приоритет, если у соревнования несколько тем.
THEMES = {
    "time series": ["time series", "forecasting"],
    "nlp": ["nlp", "text", "language", "translation"],
    "cv": ["image", "computer vision", "video", "object detection", "segmentation"],
    "tabular": ["tabular"],
}


def tag_theme(tag_name: str) -> str | None:
    name = str(tag_name).lower()
    for theme, keywords in THEMES.items():
        if any(k in name for k in keywords):
            return theme
    return None


def competition_themes(comp_tags: pd.DataFrame) -> pd.Series:
    """CompetitionId -> тема. comp_tags — CompetitionTags, объединённая с Tags (колонка Name).

    Соревнования без подходящих тегов получают тему "other".
    """
    themed = comp_tags.assign(theme=comp_tags["Name"].map(tag_theme)).dropna(subset=["theme"])
    order = {t: i for i, t in enumerate(THEMES)}
    themed = themed.assign(prio=themed["theme"].map(order)).sort_values("prio")
    return themed.drop_duplicates("CompetitionId").set_index("CompetitionId")["theme"]
