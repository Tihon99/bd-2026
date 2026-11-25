"""Reproduce the published analysis from a prepared team features table."""
import argparse
from pathlib import Path
import pandas as pd
from .competition_report import plot_counts


def summarize(df):
    df = df.copy()
    df["team_size_band"] = df["team_size"].clip(upper=5).astype(int).astype(str).replace({"5":"5+"})
    df["experience_band"] = pd.cut(df["max_prev_comps"], [-1, 0, 2, 5, 10, float("inf")],
        labels=["0", "1-2", "3-5", "6-10", "11+"])
    df["first_week_submissions_band"] = pd.cut(df["n_subs_first7d"], [-1, 0, 1, 5, 10, 25, 100, float("inf")],
        labels=["0", "1", "2-5", "6-10", "11-25", "26-100", "101+"])
    parts = []
    for feature, col in [("team_size", "team_size_band"), ("experience", "experience_band"), ("first_week_submissions", "first_week_submissions_band")]:
        g = df.groupby(col, observed=True)["top10"].agg(teams="size", top10_teams="sum", top10_rate="mean").reset_index()
        g.insert(0, "feature", feature)
        parts.append(g.rename(columns={col: "band"}))
    summary = pd.concat(parts, ignore_index=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    result = summarize(pd.read_parquet(args.dataset))
    result.to_csv(args.out / 'team_success_by_band.csv', index=False, float_format='%.6f')


if __name__ == '__main__':
    main()
