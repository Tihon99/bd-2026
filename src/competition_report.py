"""Build the competition tables and standalone figures from a Meta Kaggle snapshot."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .topics import competition_themes

SEGMENTS = ("Featured", "Research", "Playground", "Getting Started")
LABELS = {
    "cv": "Компьютерное зрение",
    "nlp": "Обработка естественного языка",
    "tabular": "Табличные данные",
    "time series": "Временные ряды",
    "no_match": "Есть теги, нет совпадений",
    "no_tags": "Нет записей о тегах",
}


def summarize(competitions, links, tags):
    """Return counts for the snapshot and mutually exclusive themes for the four segments."""
    frame = competitions.copy()
    if frame["Id"].isna().any() or frame["Id"].duplicated().any():
        raise ValueError("Competition IDs must be present and unique")
    frame["year"] = pd.to_datetime(frame["EnabledDate"], format="mixed", errors="coerce").dt.year.astype("Int64")
    frame["segment"] = frame["HostSegmentTitle"].fillna("Без типа")
    selected = frame[frame["segment"].isin(SEGMENTS)].copy()
    linked = links.merge(tags, on="TagId", how="left", validate="many_to_one")
    matched = competition_themes(linked)
    selected["theme"] = selected["Id"].map(matched)
    has_links = selected["Id"].isin(links["CompetitionId"])
    selected.loc[selected["theme"].isna() & has_links, "theme"] = "no_match"
    selected["theme"] = selected["theme"].fillna("no_tags")
    return {
        "segments": frame.groupby("segment").size().sort_values(ascending=False),
        "years": selected.groupby(["year", "segment"], dropna=False).size().unstack(fill_value=0).reindex(columns=SEGMENTS, fill_value=0),
        "themes": selected["theme"].value_counts().reindex(LABELS, fill_value=0),
        "early_dates": frame.loc[frame["year"].lt(2010), ["Id", "segment", "EnabledDate"]],
        "dates": pd.Series({"missing_or_invalid": int(frame["year"].isna().sum()), "snapshot_rows": len(frame), "selected_rows": len(selected)}),
    }


def plot_counts(counts, title, note, path):
    fig, ax = plt.subplots(figsize=(10.8, 5.1))
    fig.subplots_adjust(left=.37, right=.94, top=.79, bottom=.22)
    bars = ax.barh(counts.index, counts.values, color="#19776f")
    ax.invert_yaxis()
    ax.bar_label(bars, padding=5, fmt="%d")
    ax.set_xlim(0, max(1, counts.max()) * 1.18)
    ax.set_xlabel("Число соревнований")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", alpha=.2)
    ax.set_axisbelow(True)
    fig.text(.04, .93, title, fontsize=18, weight="bold")
    fig.text(.04, .86, note, fontsize=10)
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--snapshot", required=True, help="Source snapshot label for figures")
    args = parser.parse_args(argv)
    competitions = pd.read_csv(args.data / "Competitions.csv", usecols=["Id", "HostSegmentTitle", "EnabledDate"])
    links = pd.read_csv(args.data / "CompetitionTags.csv", usecols=["CompetitionId", "TagId"])
    tags = pd.read_csv(args.data / "Tags.csv", usecols=["Id", "Name"]).rename(columns={"Id": "TagId"})
    tables = summarize(competitions, links, tags)
    args.out.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(args.out / f"{name}.csv")
    source = f"Meta Kaggle, {args.snapshot}"
    plot_counts(tables["segments"], "Типы соревнований во всей выгрузке",
                f"{source} · {len(competitions):,} записей".replace(",", " "), args.out / "segments.png")
    plot_counts(tables["themes"].rename(index=LABELS), "Темы соревнований четырёх выбранных типов",
                f"{source} · одна тема на соревнование · всего {tables['themes'].sum()}", args.out / "themes.png")
    years = tables["years"].dropna()
    fig, ax = plt.subplots(figsize=(10.8, 5.1))
    fig.subplots_adjust(top=.78, bottom=.19)
    years.plot.bar(stacked=True, ax=ax, color=["#19776f", "#3565a4", "#da9140", "#7f5699"])
    ax.set(xlabel="Год запуска по полю EnabledDate", ylabel="Число соревнований")
    ax.legend(title="Тип в Meta Kaggle", ncol=2)
    fig.text(.06, .93, "Запуски соревнований по годам", fontsize=18, weight="bold")
    fig.text(.06, .86, f"{source} · четыре выбранных типа · последний год может быть неполным", fontsize=10)
    fig.savefig(args.out / "years.png", dpi=160, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
