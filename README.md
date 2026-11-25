# Анализ соревнований Kaggle по Meta Kaggle

Проект исследует состав и темы соревнований, результаты команд и качество модели для определения попадания в верхние 10% итогового рейтинга.

- [Какие соревнования вошли в анализ](docs/COMPETITION_FINDINGS%20%281%29.md)
- [Отчёт для офлайн-чтения](reports/review/report.pdf)
- [Слайды для офлайн-просмотра](reports/review/slides.pdf)

## Воспроизведение

Установите пакет командой `python -m pip install .` (Python 3.10 или новее). Подготовьте CSV по [инструкции](data/README.md). Запуск из корня репозитория:

```bash
python -m src.competition_report --data data/raw --out output --snapshot 2026-11-06
python -m src.build_dataset
python -m src.team_report --dataset data/processed/teams_dataset.parquet --out output
python -m src.model_report --dataset data/processed/teams_dataset.parquet --out output
python -m unittest discover -s tests -v
```

Укажите фактическую дату получения CSV в `--snapshot`. Тесты используют искусственные примеры из `tests/fixtures` и работают без Kaggle. При наличии модуля `build` команда `python -m build` собирает дистрибутив.

Ноутбуки сохраняют историю исследования. Результаты из отчёта воспроизводятся Python-командами выше. Сырые таблицы и кеш в Git не хранятся.
