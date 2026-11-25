"""Render the offline review handout and slides from the saved analysis tables."""
import argparse
from pathlib import Path
from textwrap import fill

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd

INK = "#172633"
TEAL = "#19776f"


def page(pdf, title, paragraphs, number, size, image=None):
    slide = size[0] > size[1]
    fig = plt.figure(figsize=size, facecolor="#ffffff")
    fig.text(.07, .94, title, fontsize=25 if slide else 21, color=INK, weight="bold", va="top")
    y = .82 if slide else .85
    for paragraph in paragraphs:
        wrapped = fill(paragraph, width=95 if slide else 71)
        fig.text(.07, y, wrapped, fontsize=15 if slide else 12, color=INK,
                 va="top", linespacing=1.45)
        y -= (wrapped.count("\n") + 1) * (.050 if slide else .032) + .033
    if image:
        bottom = .11
        ax = fig.add_axes([.07, bottom, .86, max(.20, y - bottom)])
        ax.imshow(plt.imread(image))
        ax.axis("off")
    fig.text(.07, .055, "Источник данных: Meta Kaggle, выгрузка от 6 ноября 2026 года", fontsize=9, color="#475961")
    fig.text(.93, .055, str(number), ha="right", fontsize=9, color=TEAL)
    pdf.savefig(fig)
    plt.close(fig)


def render(path):
    team = pd.read_csv(path / "team_success_by_band.csv")
    size = team.loc[team["feature"].eq("team_size")].copy()
    scores = pd.read_csv(path / "model_stability_by_split.csv")
    averages = scores.groupby("model")["roc_auc"].mean()
    low, high = scores.loc[scores["model"].eq("lightgbm"), "roc_auc"].agg(["min", "max"])
    first, last = size.iloc[0], size.iloc[-1]
    for name, labels, values, axis_label, limit, scale in [
        ("team_size", size["band"].astype(str), size["top10_rate"] * 100,
         "Доля команд в верхних 10% рейтинга, %", 47, "%.1f"),
        ("model_auc", ["Логистическая регрессия", "LightGBM"],
         [averages["logreg"], averages["lightgbm"]], "Средний ROC-AUC по пяти разбиениям", 1, "%.3f"),
    ]:
        fig, ax = plt.subplots(figsize=(10, 3.2), layout="constrained")
        bars = ax.barh(labels, values, color=TEAL)
        ax.bar_label(bars, fmt=scale, padding=6, fontsize=12)
        ax.set(xlim=(0, limit), xlabel=axis_label)
        if name == "team_size":
            ax.set_ylabel("Участников в команде")
        else:
            ax.axvline(.5, color="#647480", linestyle="--", label="Случайный порядок: 0,5")
            ax.legend(loc="lower right")
        ax.spines[["right", "top"]].set_visible(False)
        fig.savefig(path / f"{name}.png", dpi=160)
        plt.close(fig)
    pct = lambda value: f"{100 * value:.1f}".replace(".", ",") + "%"
    dec = lambda value: f"{value:.3f}".replace(".", ",")
    team_text = (
        f"В выборке 458 878 команд из 303 соревнований Featured. Верхние 10% определены по "
        "приватному итоговому рейтингу: место делят на число команд соревнования. "
        f"Доля составила {pct(first.top10_rate)} среди {first.teams:_} одиночных команд и "
        f"{pct(last.top10_rate)} среди {last.teams:_} команд из пяти и более человек. "
        "Сравнение описательное. На величину разницы могут влиять опыт участников и различия между соревнованиями."
    ).replace("_", " ")
    model_text = (
        "На каждом из пяти разбиений все команды одного соревнования относятся только к обучению "
        "или только к проверке. В каждом проверочном наборе 61 соревнование. ROC-AUC показывает, "
        "насколько хорошо модель ставит команды из верхних 10% выше остальных."
    )
    model_result = (
        f"Средний ROC-AUC LightGBM составил {dec(averages['lightgbm'])}, "
        f"логистической регрессии — {dec(averages['logreg'])}. "
        f"У LightGBM значения по разбиениям лежали между {dec(low)} и {dec(high)}. "
        "Оценка на соревнованиях, проведённых позже обучающих, здесь не выполнялась."
    )
    scope = "В выгрузке 12 794 уникальные записи семи типов. В динамику и разметку тем вошли 706 соревнований четырёх типов: Featured, Research, Playground и Getting Started."
    date = "Среди выбранных записей год запуска лежит между 2010 и 2026. Две более ранние даты (2000 и 2001) есть у Community. Эти записи исключены по типу; достоверность дат отдельно не проверялась."
    themes = "Темы определены поиском ключевых слов в названиях тегов. При совпадении приоритет у временных рядов, затем у языка, зрения и табличных данных. У 109 записей теги есть, но совпадений нет. У 184 записей тегов нет."
    method = "Анализ запускается командами Python. Три исходных CSV используются для анализа соревнований; для модели и команд используется таблица признаков, построенная из сырых данных. Автотесты на малых искусственных примерах проверяют отбор, даты, темы и знаменатели."
    limits = "Охват 2026 года заканчивается датой выгрузки. Тематическая разметка зависит от словаря. В отчёт включены результаты анализа соревнований, команд и модели."
    with PdfPages(path / "report.pdf") as pdf:
        page(pdf, "Анализ соревнований Kaggle", [scope, date, "Цель работы — описать соревнования, оценить наблюдаемые различия между командами и сравнить модели ранжирования.", method], 1, (8.27, 11.69), path / "segments.png")
        page(pdf, "Как определены темы", [themes, "Компьютерное зрение охватывает изображения и видео, обработка естественного языка — текст, табличные задачи — строки и столбцы, временные ряды — наблюдения во времени.", limits], 2, (8.27, 11.69), path / "themes.png")
        page(pdf, "Результаты команд и модели", [team_text, model_text, model_result, "Код, таблицы и команды запуска собраны в репозитории github.com/Tihon99/bd-2026."], 3, (8.27, 11.69))
        page(pdf, "Число запусков по годам", ["Столбец показывает число соревнований, запущенных в конкретном году по полю EnabledDate. Цвета показывают четыре выбранных типа. Данные 2026 года охватывают часть года."], 4, (8.27, 11.69), path / "years.png")
    with PdfPages(path / "slides.pdf") as pdf:
        for index, (title, body, image) in enumerate([
            ("Анализ соревнований Kaggle", [scope, "В работе сравниваются типы и темы соревнований, доли команд в верхних 10% рейтинга и качество моделей ранжирования."], None),
            ("Какие соревнования вошли в анализ", [date], "segments.png"),
            ("Как распределены темы", [themes], "themes.png"),
            ("Как менялось число запусков", ["Год определён полем EnabledDate. Цвета показывают четыре выбранных типа. Данные за 2026 год заканчиваются датой выгрузки."], "years.png"),
            ("Размер команды и итоговый рейтинг", [team_text], "team_size.png"),
            ("Модель проверена на новых соревнованиях", [model_text, model_result], "model_auc.png"),
            ("Откуда взялись числа", [method, limits], None),
        ], 1):
            page(pdf, title, body, index, (12.8, 7.2), path / image if image else None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    render(parser.parse_args().results)
