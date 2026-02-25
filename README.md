# SPPPy — Surface Plasmon Polariton Python Library

Python-библиотека для моделирования поверхностного плазмонного резонанса (SPR). Используется для расчёта коэффициентов отражения и пропускания многослойных оптических структур.

## Возможности

- Расчёт коэффициента отражения R(θ) — зависимость от угла падения
- Расчёт коэффициента отражения R(λ) — спектральная зависимость
- Поддержка градиентных слоёв с переменным показателем преломления
- Модели дисперсии: Коши, Лорентц-Друде
- Анизотропные слои
- Встроенная база оптических констант материалов
- GUI-приложение для визуализации кривых отражения

## Установка

```bash
pip install -r requirements.txt
```

Клонируйте репозиторий и используйте:

```python
import SPPPy
```

## Быстрый пример

```python
from SPPPy import ExperimentSPR, Layer, nm

# Создание эксперимента
exp = ExperimentSPR(polarization='p')
exp.wavelength = 632.8 * nm  # длина волны (HeNe лазер)

# Добавление слоёв: (показатель преломления, толщина, название)
exp.add(Layer(1.0, 0, "Air"))          # полубесконечный слой сверху
exp.add(Layer(1.52, 1000 * nm, "Glass")) # стеклянная подложка
exp.add(Layer(0.18 + 3.5j, 50 * nm, "Gold")) # золотой слой
exp.add(Layer(1.33, 0, "Sample"))       # полубесконечный слой снизу

# Расчёт отражения
angles = [30, 40, 50, 60, 70, 80]  # углы в градусах
R = exp.R(angle_range=angles)

print(f"Угол(°)\tR")
for a, r in zip(angles, R):
    print(f"{a}\t{r:.4f}")
```

## Структура проекта

```
SPPPy2026/
├── SPPPy/                    # Библиотека
│   ├── __init__.py           # Экспорт модулей
│   ├── experiment.py         # Класс ExperimentSPR
│   └── materials.py          # Слои, дисперсия, матрицы
├── gui/                      # GUI-приложение
│   ├── layer.py              # Редактор параметров слоёв
│   ├── layers_panel.py       # Панель управления слоями
│   ├── plots.py              # Вкладки с графиками
│   ├── dialogs.py            # Диалоговые окна
│   └── precision_input.py   # Виджет ввода чисел
├── app.py                    # Главное окно приложения
├── requirements.txt          # Зависимости
└── PermittivitiesBase.csv    # База оптических констант
```

## Описание модулей

### SPPPy/experiment.py

**Класс `ExperimentSPR`** — основной класс для численного расчёта SPR-экспериментов.

Основные атрибуты:
- `layers` — список слоёв структуры
- `wavelength` — длина волны излучения (в метрах)
- `incidence_angle` — угол падения (в радианах)
- `polarization` — поляризация ('p' или 's')
- `gradient_resolution` — разрешение для градиентных слоёв

Основные методы:
```python
exp.add(layer)           # Добавить слой
exp.delete(num)          # Удалить слой по индексу
exp.R(angle_range, wl_range, angle)  # Рассчитать отражение
exp.T(angle_range, wl_range, angle)  # Рассчитать пропускание
exp.save_scheme()        # Сохранить текущую схему
exp.load_scheme()        # Загрузить сохранённую схему
```

### SPPPy/materials.py

**Класс `Layer`** — контейнер для параметров оптического слоя.

```python
Layer(n, thickness, name=None)
```

Параметры:
- `n` — показатель преломления (число, дисперсия или функция)
- ` thickness` — толщина слоя (в метрах)
- `name` — название слоя (опционально)

**Класс `DispersionABS`** (абстрактный) — базовый класс для моделей дисперсии.

**Класс `CauchyDispersion`** — модель Коши для диэлектриков:
```python
cauchy = CauchyDispersion(A=1.5, B=0.005)  # n(λ) = A + B/λ²
```

**Класс `LorentzDrudeDispersion`** — модель Лорентц-Друде для металлов:
```python
ld = LorentzDrudeDispersion(ωp=9.0, Γ=0.07, f=...)
```

**Класс `MaterialDispersion`** — дисперсия из базы данных:
```python
gold = MaterialDispersion("Au")  # Золото из базы
```

**Класс `Anisotropic`** — анизотропный слой с разными показателями по осям.

**Функция `M_matrix`** — расчёт матрицы переноса между слоями.

### GUI-приложение

Запуск GUI:
```bash
python app.py
```

Возможности GUI:
- Визуализация кривых отражения R(θ), R(λ)
- Редактирование параметров слоёв
- Сохранение и загрузка схем
- Просмотр профиля градиентных слоёв

## Зависимости

См. файл [requirements.txt](requirements.txt)

## Лицензия

MIT License

## Авторы

Библиотека разработана для расчёта SPR-экспериментов в оптических системах.

## Performance baseline (API)

Ниже зафиксирована официальная спецификация baseline для сравнения производительности между PR.

### Определение KPI `curves/sec`

- **Scenario A**: 1 curve = **одно значение** `R` для фиксированных `(θ, λ)` при одном значении толщины (baseline-цикл через `R_deg`).
- **Scenario A_FAST**: 1 curve = **одно значение** `R` для фиксированных `(θ, λ)` при одном значении толщины (оптимизированный путь через `R_vs_thickness`).
- **Scenario B**: 1 curve = **полный массив** `R(λ)` по всей λ-сетке для одного значения толщины.
- **Scenario C**: 1 curve = **полный массив** `R(θ)` по всей θ-сетке для одного значения толщины.

`curves/sec = curves_count / mean_seconds_per_benchmark_call`.

### Эталонные сетки по умолчанию

- `θ`: 40°..70°, шаг 0.1°
- `λ`: 400..700 нм, шаг 1 нм
- `thickness`: 0..100 нм, шаг 1 нм

Структура в baseline: **Prism / Ag(55 нм) / SiO2(variable) / Air**.

### Режим измерений

- single-thread baseline
- reference-режим (детальный): warmup **5**, measured rounds **30**
- CI smoke-режим (быстрый): warmup **2**, measured rounds **10**
- статистика в отчётах: mean, median, stddev, p95 (из данных pytest-benchmark)

По умолчанию тесты используют reference-режим, но параметры можно переопределять переменными
`PERF_BENCH_WARMUP_ROUNDS` и `PERF_BENCH_ROUNDS`.

### Фиксация машины CI для сопоставимости

Для baseline используем закреплённый раннер:

- `ubuntu-22.04` (GitHub-hosted)
- Python `3.11`
- без параллельного запуска benchmark-job в рамках workflow

Workflow `perf-baseline` запускается автоматически на `pull_request` (для изменений в коде/тестах baseline) и вручную через `workflow_dispatch`. Это делает сравнение между PR более сопоставимым.

Чтобы не перегружать PR-пайплайн, в CI workflow используется smoke-конфигурация сеток:

- `PERF_THETA_STEP_DEG=0.2`
- `PERF_WL_STEP_NM=2.0`
- `PERF_H_STEP_NM=2.0`
- `PERF_BENCH_WARMUP_ROUNDS=2`
- `PERF_BENCH_ROUNDS=10`

Для более точного сравнения можно запускать reference-конфигурацию вручную (`workflow_dispatch`) с более плотной сеткой и 5/30.

См. workflow: `.github/workflows/perf-baseline.yml`.

> Важно: локальные замеры на Windows 11 / Python 3.12 полезны для локального анализа, но не стоит напрямую сравнивать их с CI baseline из-за различий ОС, интерпретатора и железа. Сравнение «до/после» для PR лучше делать по CI-артефактам.

### Запуск baseline и экспорт JSON + CSV

```bash
RUN_PERF_BASELINE=1 pytest tests/test_performance_baseline.py \
  --benchmark-only \
  --benchmark-json artifacts/benchmark/benchmark.json

python tests/benchmark_to_csv.py \
  --benchmark-json artifacts/benchmark/benchmark.json \
  --csv-output artifacts/benchmark/benchmark_summary.csv
```

Можно переопределять сетки через переменные окружения:

- `PERF_THETA_START_DEG`, `PERF_THETA_STOP_DEG`, `PERF_THETA_STEP_DEG`
- `PERF_WL_START_NM`, `PERF_WL_STOP_NM`, `PERF_WL_STEP_NM`
- `PERF_H_START_NM`, `PERF_H_STOP_NM`, `PERF_H_STEP_NM`

### Двухступенчатое профилирование

```bash
python tests/profile_baseline.py --output-dir artifacts/profiling
```

- Stage 1: `cProfile` + `pstats` (top cumulative)
- Stage 2: `line_profiler` для `Transfer_matrix`, `Layer.S_matrix`, `DispersionABS.CRI` (если пакет установлен)

### Управление глобальным `M_cache` (фаза оптимизаций)

Добавлены API-хелперы для контроля глобального кеша межслойных матриц:

- `SPPPy.set_m_cache_limit(maxsize)`
- `SPPPy.get_m_cache_limit()`
- `SPPPy.get_m_cache_size()`
- `SPPPy.clear_m_cache()`

Текущая политика: при достижении лимита кеш полностью очищается (`clear-on-limit`).
