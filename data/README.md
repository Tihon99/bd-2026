# Данные

Источник: [Meta Kaggle](https://www.kaggle.com/datasets/kaggle/meta-kaggle) — официальный
датасет Kaggle, обновляется ежедневно. CSV-файлы кладутся в `data/raw/` и в git не попадают.

## Как скачать

1. Получить API-токен: Kaggle → Settings → API → Create New Token,
   положить `kaggle.json` в `~/.kaggle/` (Windows: `C:\Users\<имя>\.kaggle\`).
2. Запустить из корня репозитория:

```bash
bash scripts/download_data.sh
```

Скрипт скачивает только нужные таблицы, весь датасет не нужен.
Submissions — самая большая таблица, её читаем по частям через `iter_chunks`.

## Какие таблицы используем

| Таблица | Что в ней | Блок |
|---|---|---|
| Competitions | соревнования: тип, даты, призы, число команд | 1, 3 |
| Tags, CompetitionTags | теги соревнований (тематика) | 1 |
| Users | пользователи, уровень (PerformanceTier), дата регистрации | 2, 5 |
| Teams | команды, итоговое место, медаль | 2, 3, 5 |
| TeamMemberships | кто в какой команде | 2, 3, 5 |
| Kernels, KernelTags | ноутбуки: просмотры, голоса, теги | 4 |
| Submissions | посылки команд (очень большая, читаем по частям) | 3, 5 |

`PerformanceTier`: 0 — Novice, 1 — Contributor, 2 — Expert, 3 — Master, 4 — Grandmaster, 5 — Staff.
