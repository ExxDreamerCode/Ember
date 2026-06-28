<p align="center">
  <img src="logo.png" alt="Ember Logo" width="200">
</p>

# 🔥 Ember — шахматный движок на Rust

<p align="center">
  <img src="https://img.shields.io/badge/rust-1.70%2B-orange" alt="Rust Version">
  <img src="https://img.shields.io/badge/UCI-compatible-brightgreen" alt="UCI Compatible">
  <img src="https://img.shields.io/badge/license-MIT-blue" alt="License">
</p>

**Ember** — это UCI-совместимый шахматный движок на Rust, который я пишу для изучения и экспериментов. Проект в активной разработке, регулярно допиливается и улучшается.

## 📋 Требования

- **Rust** 1.70 или новее
- UCI-совместимая оболочка (например, [Arena](http://www.playwitharena.com/), [Cute Chess](https://cutechess.com/), [Lichess](https://lichess.org/))

## 🔧 Установка

- Скачайте [последний релиз](https://github.com/ExxDreamerCode/Ember/releases/tag/V1.1.0)

## ♟️ Использование

### С графической оболочкой

1. Откройте вашу UCI-совместимую шахматную программу
2. Добавьте движок: укажите путь к скачанному бинарнику
3. Начинайте игру!

### Командная строка

```bash
# Интерактивный режим
cargo run --release

# Или передача UCI-команд
echo -e "uci\nisready\nquit" | cargo run --release
```

### UCI-опции

| Опция       | Тип    | По умолч.    | Диапазон | Описание                          |
|-------------|--------|--------------|----------|-----------------------------------|
| `Hash`      | spin   | 128          | 1–4096   | Размер TT в мегабайтах            |
| `Threads`   | spin   | 1            | 1-256        | Количество потоков     |
| `Book`      | string | `<embedded>` | —        | Путь к дебютной книге .bin        |
| `NNUE`      | string | `<embedded>` | —        | Путь к файлу нейросети .nnue      |
| `TraceFile` | string | `<empty>`    | —        | Путь к TraceBack файлу .jsonl     |
| `SyzygyPath` | string | `<empty>` | — | Путь к папке с Syzygy таблицами (DTZ) |
| `UCI_Chess960` | string | `false`    | —        | Включение/отключение Chess 960     |

### Дебютная книга

Движок поддерживает Polyglot-формат дебютных книг (.bin). В бинарник **встроена** книга по умолчанию — она загружается автоматически, если `book.bin` не найден рядом с исполняемым файлом.

Приоритет загрузки:
1. `book.bin` рядом с исполняемым файлом
2. `book.bin` в текущей рабочей папке
3. **Встроенная книга** (если внешняя не найдена)

Можно указать путь к книге через UCI:

```
setoption name Book value C:\путь\к\book.bin
```

Если книга лежит в одной папке с движком, достаточно имени файла:

```
setoption name Book value book.bin
```

Чтобы **отключить** книгу — передать пустое значение:

```
setoption name Book value
```

Чтобы **вернуться** к встроенной книге:

```
setoption name Book value <embedded>
```

Поддерживаются любые Polyglot-совместимые книги (например, от Stockfish).

### Нейросеть (NNUE)

В бинарник **встроена** NNUE-сеть — она загружается автоматически при старте. 
Внешний файл `net.nnue` рядом с исполняемым файлом **не требуется**.

По умолчанию используется встроенная сеть. Управление через UCI-опцию `NNUE`:

```
setoption name NNUE value                  # отключить NNUE (фолбэк на классический eval)
setoption name NNUE value <embedded>        # вернуться к встроенной сети
setoption name NNUE value C:\путь\к\file.nnue  # загрузить внешнюю сеть
```

Если файл лежит рядом с движком, можно указать только имя:

```
setoption name NNUE value my-net.nnue
```

При загрузке внешней сети движок выведет информацию о её версии и архитектуре:

```
info string Loaded NNUE v6 my-net.nnue SCReLU (FT=1024 L1=0 L2=0)
```

## ⚙️ Настройка

Изменение параметров движка через UCI-команду `setoption`:

```
setoption name Hash value 256
setoption name Book value book.bin
setoption option name TraceFile value Trace.jsonl
```

## 📊 Измерение Elo

В репозитории есть Nix-окружение и скрипты для автоматического прогона
матчей через Cute Chess:

```bash
nix develop .#elo-runner
python3 tools/measure_elo.py all --config configs/elo/default.toml
```

Для более точной оценки силы против Stockfish можно использовать
адаптивный режим. Он сначала играет короткий пилотный матч на нескольких
значениях `UCI_Elo`, затем выбирает уровень Stockfish, близкий к 50%
результата Ember, и тратит оставшийся бюджет игр на этот уровень:

```bash
python3 tools/measure_elo.py all \
  --config configs/elo/stockfish-adaptive.toml \
  --max-games 500
```

`--max-games` задаёт верхнюю границу числа игр, которые можно запланировать.
Результат адаптивного режима — это `Stockfish UCI_Elo equivalent` для
данного контроля времени, книги и набора дебютов, а не точная внешняя
CCRL-оценка.

Ориентировочная цена точности ниже рассчитана для 8-ядерного CPU, где
автоматический режим использует `ceil(8 * 1.5) = 12` workers, контроль
времени `8+0.08`, и соперник подобран так, чтобы результат был около 50%.

| 95% CI | Ширина интервала | Игры | Примерное время |
| ---: | ---: | ---: | ---: |
| ±50 Elo | 100 Elo | ~185 | ~8-12 мин |
| ±40 Elo | 80 Elo | ~290 | ~13-18 мин |
| ±30 Elo | 60 Elo | ~515 | ~22-33 мин |
| ±20 Elo | 40 Elo | ~1 160 | ~50-75 мин |
| ±15 Elo | 30 Elo | ~2 060 | ~1.5-2.2 ч |
| ±10 Elo | 20 Elo | ~4 640 | ~3.3-5 ч |
| ±7.5 Elo | 15 Elo | ~8 250 | ~6-9 ч |
| ±5 Elo | 10 Elo | ~18 550 | ~13-20 ч |

По нашим данным рейтинг новейшей версии Ember **2852** +- 30 elo (В однопоточном режиме)

## 📈 Замер формы поиска

Для проверки регрессий, где важны не только NPS, но и достигнутая глубина,
число узлов и форма дерева, есть отдельный UCI-бенч:

```bash
cargo build --release
nix run .#search-shape-benchmark -- \
  current=./target/release/ember \
  --repeats 3
```

Можно сравнивать несколько бинарников одним запуском:

```bash
nix run .#search-shape-benchmark -- \
  good=/path/to/good-ember \
  bad=/path/to/bad-ember \
  --repeats 3 \
  --go-command "go depth 20"
```

По умолчанию скрипт отключает книгу через `setoption name Book value`,
использует стартовую позицию, `Hash=64` и `Threads=1`. Для своего набора
позиций можно передать JSON-файл через `--positions`.

## 🛠️ Разработка

```bash
# Запуск тестов
cargo test

# Проверка ошибок
cargo check

# Запуск с оптимизациями
cargo run --release

# Компиляция в релиз - режиме
cargo build --release
```

## 🤝 Вклад

Нашёлся баг? Есть идея? Открывайте issue или PR — буду рад помощи и обратной связи!

## 📄 Лицензия

Этот проект распространяется под лицензией MIT.
