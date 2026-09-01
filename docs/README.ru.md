<p align="center">
  <img src="../logo.png" alt="Ember Logo" width="200">
</p>

# 🔥 Ember — шахматный движок на Rust

<p align="center">
  <img src="https://img.shields.io/badge/rust-nightly--2026--02--08%2B-orange" alt="Rust Version">
  <img src="https://img.shields.io/badge/UCI-compatible-brightgreen" alt="UCI Compatible">
  <img src="https://img.shields.io/badge/license-MIT-blue" alt="License">
</p>

**Ember** — это UCI-совместимый шахматный движок на Rust, который я пишу для изучения и экспериментов. Проект в активной разработке, регулярно допиливается и улучшается.

## 📋 Требования

- **Rust nightly** для сборки из исходников. Используйте закреплённый toolchain
  из `rust-toolchain.toml` (`nightly-2026-07-07`). Самый старый nightly,
  который был проверен на этой ревизии, — `nightly-2026-02-08`.
- UCI-совместимая оболочка (например, [Arena](http://www.playwitharena.de/), [Cute Chess](https://cutechess.com/), [Lichess](https://lichess.org/))

## 🔧 Установка

- Скачайте [последний релиз](https://github.com/ExxDreamerCode/Ember/releases/latest)

Подробная инструкция по воспроизводимым Nix-сборкам, выпускным архивам и
portable Windows bundle вынесена в [BUILD.md](../BUILD.md).

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
| `Hash`      | spin   | 256          | 1–4096   | Размер TT в мегабайтах            |
| `Threads`   | spin   | 1            | 1-256        | Количество потоков     |
| `Book`      | string | `<embedded>` | —        | Путь к дебютной книге .bin        |
| `RandomBookMove` | check | false | — | Равновероятно выбирать среди надёжных ходов из книги в пределах 5 сантипешек от лучшей статической оценки |
| `BookMinMoveWeight` | spin | 2 | 1-65535 | Минимальный абсолютный вес хода из книги |
| `BookMinMoveWeightPermille` | spin | 10 | 0-1000 | Минимальная доля веса хода в промилле |
| `NNUE`      | string | `<embedded>` | —        | Путь к файлу нейросети .nnue      |
| `NNUEBackend` | combo | `auto` | `auto`, доступные backend-ы | Backend для NNUE-поиска Ember V1/V2 |
| `TraceFile` | string | `<empty>`    | —        | Путь к TraceBack файлу .jsonl     |
| `SyzygyPath` | string | `<empty>` | — | Путь к папке с Syzygy таблицами (DTZ) |
| `UCI_Chess960` | check | `false`    | —        | Включение/отключение Chess 960     |
| `Tune` | string | `<empty>` | — | Оверрайды тюнингуемых констант поиска на лету (см. [docs/auto-tuning.md](auto-tuning.md)) |

### Syzygy через Nix

Репозиторий содержит Nix-цель для полного набора Syzygy 3-4-5 WDL+DTZ
из зеркала Lichess:

```
nix build .#syzygy
```

Все 290 файлов скачиваются как fixed-output derivations с SHA-256 из
`nix/syzygy-3-4-5.json`. Получившийся путь можно передать движку:

```
setoption name SyzygyPath value ./result/share/syzygy/3-4-5
```

Это набор до 5 фигур, размером 983957920 байт. Алиас `syzygy` намеренно
остаётся этим небольшим набором.

Для полного набора до 6 фигур используйте отдельную цель:

```
nix build .#syzygy-6
setoption name SyzygyPath value ./result/share/syzygy/3-4-5-6
```

`syzygy-6` (также доступен как `syzygy-3-4-5-6`) объединяет таблицы 3-5
и 6 фигур в одной папке, чтобы переходы после взятия также можно было
пробовать через Syzygy. Набор содержит 1020 файлов и занимает
161209573952 байта (около 150 GiB), поэтому для Nix store понадобится
значительный запас свободного места. SHA-256 и размеры 6-фигурных файлов
зафиксированы в `nix/syzygy-6.json`; манифест можно воспроизвести скриптом
`nix/generate-syzygy-manifest.py` из метаданных зеркала Lichess.

### Дебютная книга

Движок поддерживает Polyglot-формат дебютных книг (.bin). В бинарник
**встроена** книга по умолчанию — она загружается автоматически при запуске.

Ember не ищет `book.bin` рядом с исполняемым файлом или в текущей рабочей
папке без команды. Внешняя книга используется только после явной настройки UCI
`Book`.

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

Backend для NNUE-поиска выбирается автоматически по процессору и одинаково
применяется к сетям Ember V1 и Ember V2. Для проверок и замеров его можно
переопределить:

```
setoption name NNUEBackend value scalar
setoption name NNUEBackend value x86-v3
setoption name NNUEBackend value x86-avx512
setoption name NNUEBackend value aarch64-simd512
setoption name NNUEBackend value auto
```

Недоступный на текущем процессоре backend будет проигнорирован.

При загрузке внешней сети движок выведет информацию о её версии и архитектуре:

```
info string Loaded NNUE v6 my-net.nnue SCReLU (FT=1024 L1=0 L2=0)
```

### Поддерживаемые архитектуры NNUE

Формат сети определяется автоматически по заголовку файла. Опция `NNUE`
поддерживает следующие архитектуры:

- **Ember V1** — родной формат `.nnue`. Фича-экстрактор на основе
  короля с king-bucket-входом, опциональными threat-фичами, активациями
  `CReLU`/`SCReLU`/pairwise и необязательными скрытыми слоями `L1`/`L2`. Выводится
  как `Loaded NNUE <версия> <имя> <активация> (...)`.
- **Ember V2** — формат-контейнер для второй версии архитектуры Ember с фиксированной
  раскладкой (полу-трансформер с PSQ- и threat-фичами, за которым следуют
  несколько стеков `Affine`); размеры проверяются при загрузке. Выводится как
  `Loaded Ember V2 net <путь> (arch hash=... desc="..." ...)`. Эти файлы уже
  используют сжатие из NNUE PyTorch и не требуют преобразования в компактный
  формат Ember `ECN1`.
- **Компактный формат (ECN1)** — сжатый вариант собственной сети V1 Ember (магическое
  число `ECN1`, создаётся скриптом `training/v1/compact_nnue_v1.py`). Он сокращает
  фича-трансформер до плотного базиса плюс карты коррекции — это не отдельная
  архитектура, а лишь формат хранения той же собственной сети. Через опцию `NNUE`
  загружаются как внешние файлы `ECN1`, так и встроенная сеть в этом формате.
- **Классическая сеть `HalfKP(Friend)`** — классический формат Stockfish с фичей
  `HalfKP(Friend)` и слоями `AffineTransform` + `ClippedReLU`. Выводится как
  `Loaded legacy HalfKP net <путь>`.

## ⚙️ Настройка

Изменение параметров движка через UCI-команду `setoption`:

```
setoption name Hash value 256
setoption name Book value book.bin
setoption option name TraceFile value Trace.jsonl
```

## 📊 Оценка качества

Измерение Elo, попарное сравнение версий и замер формы поиска описаны в
[quality-assessment.md](quality-assessment.md).

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
