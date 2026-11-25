import unittest
from pathlib import Path

import pandas as pd

from src.competition_report import summarize
from src.team_report import summarize as team_summary


class CompetitionReportTests(unittest.TestCase):
    def setUp(self):
        fixture = Path(__file__).parent / "fixtures"
        self.competitions = pd.read_csv(fixture / "Competitions.csv")
        self.links = pd.read_csv(fixture / "CompetitionTags.csv")
        self.tags = pd.read_csv(fixture / "Tags.csv").rename(columns={"Id": "TagId"})

    def test_selection_and_unclassified_reasons(self):
        result = summarize(self.competitions, self.links, self.tags)
        self.assertEqual(result["segments"].sum(), 6)
        self.assertEqual(result["segments"]["Без типа"], 1)
        self.assertEqual(result["themes"].sum(), 4)
        self.assertEqual(result["themes"].to_dict(), {
            "cv": 0, "nlp": 1, "tabular": 0, "time series": 0,
            "no_match": 2, "no_tags": 1,
        })

    def test_dates_preserve_unknown_and_early_rows(self):
        result = summarize(self.competitions, self.links, self.tags)
        self.assertEqual(result["early_dates"]["Id"].tolist(), [5])
        self.assertEqual(result["dates"]["missing_or_invalid"], 1)
        self.assertEqual(result["years"].loc[pd.NA, "Getting Started"], 1)

    def test_duplicate_competition_ids_fail(self):
        duplicated = pd.concat([self.competitions, self.competitions.iloc[:1]])
        with self.assertRaisesRegex(ValueError, "unique"):
            summarize(duplicated, self.links, self.tags)


class TeamReportTests(unittest.TestCase):
    def test_bands_and_denominators(self):
        teams = pd.DataFrame({
            "team_size": [5, 6, 1],
            "max_prev_comps": [0, 11, float("nan")],
            "n_subs_first7d": [0, 1, 2],
            "top10": [True, False, True],
        })
        result = team_summary(teams).set_index(["feature", "band"])
        self.assertEqual(result.loc[("team_size", "5+"), "teams"], 2)
        self.assertEqual(result.loc[("team_size", "5+"), "top10_rate"], 0.5)
        self.assertEqual(result.loc["experience", "teams"].sum(), 2)
        self.assertEqual(result.loc[("first_week_submissions", "2-5"), "top10_teams"], 1)
