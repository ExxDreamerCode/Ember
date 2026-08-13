# Об автотюнинге констант поиска

Система автотюнинга позволяет искать более сильные значения констант поиска
путём последовательных SPRT-матчей против текущего оптимального набора.
Она не перекомпилирует движок для каждого кандидата: один release-бинарник
использует единую UCI-опцию `Tune`, чтобы различать «текущий best» и
«кандидата» на лету.

## Как это устроено

### Инфраструктура в движке

Модуль `src/tune.rs` держит глобальные атомарные оверрайды для помеченных
констант. В обычных играх (когда `Tune` не задана) hot-path чтение делает один
`load(Relaxed)` и возвращает compile-time дефолт — накладных практически нет.

Константы, доступные для тюнинга, и их дефолты:

### Selectivity (src/search/selectivity.rs)

| Параметр | Дефолт | Смысл |
| --- | ---: | --- |
| `PROBCUT_MIN_DEPTH` | 8 | Минимальная глубина для ProbCut |
| `PROBCUT_MARGIN_CP` | 350 | Запас ProbCut в сантипешках |
| `ROOT_REPETITION_TIE_MIN_SCORE` | 300 | Минимальный счёт для выбора не повторяющегося хода в корне |

### Reverse futility / futility pruning (src/search/negamax.rs)

| Параметр | Дефолт | Смысл |
| --- | ---: | --- |
| `REVERSE_FUTILITY_BASE_CP` | 80 | База RFP-margin |
| `REVERSE_FUTILITY_PER_DEPTH_CP` | 65 | Margin RFP на глубину |
| `REVERSE_FUTILITY_MAX_DEPTH` | 8 | Максимальная глубина RFP |
| `FUTILITY_MARGIN_PER_DEPTH_CP` | 150 | Margin futility на глубину |
| `FUTILITY_MAX_DEPTH` | 3 | Максимальная глубина futility |

### Null move

| Параметр | Дефолт | Смысл |
| --- | ---: | --- |
| `NULL_MOVE_MIN_DEPTH` | 3 | Минимальная глубина для null move |
| `NULL_MOVE_REDUCTION_BASE` | 3 | База редукции |
| `NULL_MOVE_REDUCTION_DIVISOR` | 4 | Делитель глубины в редукции |
| `NULL_MOVE_MARGIN_DIVISOR` | 200 | Делитель (eval − beta) в редукции |
| `NULL_MOVE_MARGIN_CAP` | 3 | Потолок маржинальной части редукции |
| `NULL_MOVE_KING_PRESSURE_LIMIT` | 3 | Макс. давление на короля для null move |
| `NULL_MOVE_NON_PAWN_LIMIT` | 4 | Минимум непешечного материала для null move |

### Прюнинг (negamax.rs)

| Параметр | Дефолт | Смысл |
| --- | ---: | --- |
| `SEE_MARGIN_PER_DEPTH_CP` | 80 | Margin SEE-прюнинга на глубину |
| `HISTORY_PRUNE_MARGIN_PER_DEPTH` | 1024 | Margin history-прюнинга на глубину |
| `HISTORY_PRUNE_MAX_DEPTH` | 5 | Максимальная глубина history-прюнинга |

### Селективность (negamax.rs)

| Параметр | Дефолт | Смысл |
| --- | ---: | --- |
| `CHECK_EXTENSION_MAX_DEPTH` | 16 | Максимальная глубина для шахового расширения |
| `LMP_MAX_DEPTH` | 8 | Максимальная глубина late move pruning |
| `IID_MIN_DEPTH` | 4 | Минимальная глубина для внутренней итерации |
| `LMR_DIVISOR_MILLIS` | 1800 | Делитель LMR-редукции (ln(move)·ln(depth)·1000 / divisor) |

### UCI-интерфейс

```text
option name Tune type string default <empty>
```

Применение одного или нескольких оверрайдов:

```text
setoption name Tune value "PROBCUT_MARGIN_CP=400,PROBCUT_MIN_DEPTH=12"
```

Форматы значения `Tune`:

- CSV: `NAME=VALUE,NAME2=VALUE2` — рекомендуемый формат для cutechess,
  поскольку один `option.Tune=...` не содержит пробелов;
- пробельный: `NAME VALUE` — один параметр за вызов.

Просмотр активных оверрайдов:

```text
tune
```

Пример ответа:

```text
info string tune PROBCUT_MIN_DEPTH = 12
info string tune PROBCUT_MARGIN_CP = 400
```

Очистка:

```text
setoption name Tune value ""
```

Важно: команда `ucinewgame` **не** сбрасывает оверрайды. Это нужно, чтобы
настройка переживала переход между партиями внутри матча.

## Автотюнер

Инструменты живут в `tools/auto_tune/`:

| Файл | Назначение |
| --- | --- |
| `tune.toml` | Описание параметров, диапазонов, SPRT и дебютов |
| `seek.py` | Координатный спуск: перебирает соседей каждого параметра, гоняет SPRT против текущего best, ведёт журнал |
| `apply.py` | Показывает значения из `best.json`, отличающиеся от дефолтов, для ручного переноса в код |
| `best.json` | Текущие оптимальные значения (создаётся автоматически) |
| `journal.jsonl` | Полный журнал каждого SPRT-матча |

### Запуск

`seek.py` — обычный Python-скрипт, его не обязательно запускать внутри Nix.
Он запускает матчи через `head_to_head.py`, поэтому для реального тюнинга
нужны те же зависимости, что и у head-to-head runner'а:

- Python 3.11+ со стандартным модулем `tomllib`;
- пакет `python-chess` (см. `requirements.txt`);
- `cutechess-cli` в `PATH`;
- release-бинарник движка (аргумент `--engine`).

Всё это даёт dev-shell `nix develop .#elo-runner`, либо можно поставить
зависимости напрямую (`pip install -r requirements.txt` + `cutechess-cli`):

```bash
# Сборка release-бинарника (нужен только один раз)
cargo build --release --bin ember

# Тюнинг всех параметров с указанным контролем времени
python tools/auto_tune/seek.py --time-control 8+0.08

# Тюнинг выбранных параметров
python tools/auto_tune/seek.py --params PROBCUT_MIN_DEPTH,PROBCUT_MARGIN_CP

# Просмотр найденных значений
python tools/auto_tune/apply.py

# Репетиция без реальных матчей (не требует cutechess и бинарника)
python tools/auto_tune/seek.py --dry-run
```

Если вы работаете в Linux dev-shell `nix develop .#elo-runner`, то
`cutechess-cli`, `python-chess` и toolchain уже доступны, и `--engine`
по умолчанию `target/release/ember` соберётся там же.

### Как принимаются решения

1. Для параметра берётся текущее значение (из `best.json`, либо дефолт из
   `tune.toml`).
2. Пробуется сосед `current + step`, затем `current - step`.
3. Каждый кандидат сравнивается с текущим best через `head_to_head.py run`
   с включённым pentanomial SPRT (elo0=0, elo1=5, alpha=beta=0.05).
   `engine_a` — incumbent, `engine_b` — кандидат.
4. Кандидат принимается только когда SPRT отвергает нулевую гипотезу
   (`engine_b_better` — «candidate лучше»). Вердикт `engine_a_better`
   («incumbent лучше») означает отклонение кандидата, а
   `inconclusive`/`continue` — недостаточно данных. После принятия значение
   становится новым best и процесс повторяется в том же направлении.
5. Когда оба соседа отвергнуты, пробуется более широкий шаг `current + 2*step`.
6. Параметр замирает, когда ни один сосед не проходит — результат помечается
   «settled».

Тайм-контроли из `common.time_controls` чередуются между SPRT-матчами по
кругу (счётчик берётся из числа записей в `journal.jsonl`). Флаг
`--time-control` переопределяет набор целиком. После каждого принятого
значения `best.json` перезаписывается сразу, а не только в конце прогона,
поэтому прерванный тюнинг можно продолжить.

Все матчи записываются в `journal.jsonl`: параметр, старое/новое значение,
вердикт, принято/нет, Elo, score rate, пары/игры, LLR, SHA-256 бинарника,
time control, параметры SPRT. Это позволяет воспроизвести любой результат и
понять, почему значение было принято или отклонено.

### Отчёты о прогонах

Для каждого матча в `results/tune/<run_id>/` создаются два файла:

- **`report.md`** — человекочитаемый отчёт: вердикт (принято/отклонено/
  неопределённо), Elo (candidate − incumbent), score rate, пары/игры, LLR,
  time control, параметры SPRT, SHA-256 бинарника, время.
- **`report.json`** — машиночитаемая версия: `record` (все поля журнала) +
  `summary` (полная статистика из `head_to_head.py`).

`run_id` имеет вид `tune-<param>-<value>-<timestamp>`, поэтому каждый прогон
легко найти и сопоставить с записью в `journal.jsonl`.

### Конфигурация `tune.toml`

```toml
results_dir = "results/tune"

[common]
time_controls = ["8+0.08", "1+0.01"]   # по какому ТС играть
max_pairs = 300                        # лимит пар на один SPRT
min_pairs = 20
seed = 20260714
opening_source = "polyglot"
polyglot_book = "src/book.bin"
hash_mb = 64
threads = 1
cutechess_cmd = "cutechess-cli"

[sprt]
enabled = true
elo0 = 0
elo1 = 5
alpha = 0.05
beta = 0.05

[[params]]
name = "PROBCUT_MIN_DEPTH"
base = 8
min = 4
max = 16
step = 1
```

`seek.py` собирает временный TOML-конфиг head-to-head для каждого матча:
`engine_a` — incumbent со значениями из `best.json`, `engine_b` — кандидат,
отличающийся только тюнингуемым параметром. Остальные параметры, уже
улучшенные ранее, передаются обеим сторонам одинаково, поэтому каждый матч
измеряет только один параметр.

## Важные ограничения

- Автотюнер **не меняет код** — он лишь пишет `best.json` и `journal.jsonl`.
  Значения из `apply.py` нужно перенести в `src/` вручную и подтвердить своим
  SPRT перед коммитом.
- Рекомендуется запускать на незанятой машине и не держать параллельные
  CPU-насыщенные процессы — тайминги и NPS будут искажены.
- SPRT с `elo0=0, elo1=5` — строгий тест: небольшие улучшения могут требовать
  много пар. `max_pairs` ограничивает расход времени на бесперспективных
  кандидатов.
- Изменение значения параметра влияет на форму дерева поиска, поэтому после
  принятия значения всегда стоит прогнать обычные корректностные проверки
  (`cargo test --all-features`, `cargo clippy`) и сравнить NPS/search shape.

## Добавление нового параметра

1. Добавьте вариант в `TuneParam` в `src/tune.rs` (имя, индекс, `from_name`).
2. В месте использования константы замените чтение на
   `tune::get_int(TuneParam::NewParam, DEFAULT)`.
3. Добавьте `[[params]]` в `tools/auto_tune/tune.toml`.
4. Прогоните `cargo test --all-features` и убедитесь, что дефолтный путь
   неизменен.