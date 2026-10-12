#!/usr/bin/env bash
# Скачивает нужные таблицы Meta Kaggle в data/raw/
set -e

OUT=data/raw
TABLES="Competitions Users Teams TeamMemberships Tags CompetitionTags Kernels KernelTags"

mkdir -p "$OUT"
for t in $TABLES; do
    echo "-> $t.csv"
    kaggle datasets download -d kaggle/meta-kaggle -f "$t.csv" -p "$OUT" --unzip
done
echo "Готово: $OUT"
