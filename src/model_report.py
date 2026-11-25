"""Reproduce the published analysis from a prepared team features table."""
import argparse
from pathlib import Path
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import GroupShuffleSplit
from src.build_dataset import FEATURES, TARGET
from src.modeling import baseline_models


def summarize(df):
    df = df.copy()
    seeds = [17, 29, 42, 73, 101]
    rows = []
    for seed in seeds:
        split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        train_idx, test_idx = next(split.split(df, groups=df["CompetitionId"]))
        train, test = df.iloc[train_idx], df.iloc[test_idx]
        X_train, y_train = train[FEATURES], train[TARGET]
        X_test, y_test = test[FEATURES], test[TARGET]
        predictions = {"experience_rule": X_test["max_prev_comps"].fillna(0).to_numpy()}
        for name, model in baseline_models().items():
            model.fit(X_train, y_train)
            predictions[name] = model.predict_proba(X_test)[:, 1]
        model = LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                               is_unbalance=True, random_state=42, verbose=-1, n_jobs=4)
        model.fit(X_train, y_train)
        predictions["lightgbm"] = model.predict_proba(X_test)[:, 1]
        for name, score in predictions.items():
            rows.append({"seed": seed, "model": name,
                         "test_competitions": test["CompetitionId"].nunique(),
                         "train_rows": len(train), "test_rows": len(test),
                         "roc_auc": roc_auc_score(y_test, score),
                         "pr_auc": average_precision_score(y_test, score),
                         "test_top10_rate": y_test.mean()})

    results = pd.DataFrame(rows)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    result = summarize(pd.read_parquet(args.dataset))
    result.to_csv(args.out / 'model_stability_by_split.csv', index=False, float_format='%.6f')


if __name__ == '__main__':
    main()
