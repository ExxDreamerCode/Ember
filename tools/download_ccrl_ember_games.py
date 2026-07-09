#!/usr/bin/env python3
"""Download CCRL archive games involving Ember from one tournament archive."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen


DEFAULT_ARCHIVE_URL = "https://ccrl.live/pgns/125th_Amateur_D11/"
DEFAULT_ENGINE = "Ember 1.1.1"
DEFAULT_OUT_DIR = Path("ratings/ccrl/125th_Amateur_D11/ember_1.1.1")
USER_AGENT = "Ember-analysis-downloader/1.0"


class LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.hrefs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        for name, value in attrs:
            if name.lower() == "href" and value:
                self.hrefs.append(value)


@dataclass(frozen=True)
class SelectedGame:
    filename: str
    url: str
    meta_filename: str | None
    meta_url: str | None
    tags: dict[str, str]


def fetch_bytes(url: str, timeout: int) -> bytes:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=timeout) as response:
        return response.read()


def fetch_text(url: str, timeout: int) -> str:
    return fetch_bytes(url, timeout).decode("utf-8", errors="replace")


def archive_slug(archive_url: str) -> str:
    path = urlparse(archive_url).path.rstrip("/")
    return path.rsplit("/", 1)[-1] or "archive"


def extract_links(html: str, archive_url: str) -> list[str]:
    parser = LinkParser()
    parser.feed(html)
    links = []
    seen = set()
    for href in parser.hrefs:
        absolute = urljoin(archive_url, href)
        if absolute not in seen:
            links.append(absolute)
            seen.add(absolute)
    return links


def filename_from_url(url: str) -> str:
    return Path(urlparse(url).path).name


def parse_pgn_tags(pgn: str) -> dict[str, str]:
    tags: dict[str, str] = {}
    for line in pgn.splitlines():
        if not line.startswith("["):
            if tags:
                break
            continue
        match = re.match(r'^\[([A-Za-z0-9_]+)\s+"((?:\\.|[^"\\])*)"\]$', line)
        if not match:
            continue
        key, value = match.groups()
        tags[key] = value.replace(r"\"", '"').replace(r"\\", "\\")
    return tags


def normalized(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


def player_matches(name: str, engine: str) -> bool:
    return normalized(engine) in normalized(name)


def sort_key(filename: str) -> tuple[int, str]:
    match = re.match(r"^(\d+)_", filename)
    prefix = int(match.group(1)) if match else 10**9
    return prefix, filename


def write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def find_matching_meta(pgn_url: str, archive_links: set[str]) -> tuple[str | None, str | None]:
    meta_url = pgn_url[:-4] + ".meta.json"
    if meta_url not in archive_links:
        return None, None
    return filename_from_url(meta_url), meta_url


def crosstable_ember_games(tournament_results: dict[str, object], engine: str) -> int | None:
    parsed = tournament_results.get("parsedResults")
    if not isinstance(parsed, dict):
        return None
    standings = parsed.get("standings")
    if not isinstance(standings, list):
        return None
    for row in standings:
        if not isinstance(row, dict):
            continue
        name = row.get("name")
        games = row.get("games")
        if isinstance(name, str) and player_matches(name, engine) and isinstance(games, int):
            return games
    return None


def crosstable_ember_entries(tournament_results: dict[str, object], engine: str) -> list[dict[str, object]]:
    parsed_games = tournament_results.get("parsedGames")
    if not isinstance(parsed_games, list):
        return []
    entries = []
    for game in parsed_games:
        if not isinstance(game, dict):
            continue
        white = game.get("white")
        black = game.get("black")
        if not isinstance(white, str) or not isinstance(black, str):
            continue
        if player_matches(white, engine) or player_matches(black, engine):
            entries.append(game)
    return entries


def build_csv_rows(selected: Iterable[SelectedGame]) -> list[dict[str, str]]:
    rows = []
    for game in selected:
        rows.append(
            {
                "filename": game.filename,
                "date": game.tags.get("Date", ""),
                "white": game.tags.get("White", ""),
                "black": game.tags.get("Black", ""),
                "result": game.tags.get("Result", ""),
                "site": game.tags.get("Site", ""),
                "meta_filename": game.meta_filename or "",
                "source_url": game.url,
            }
        )
    return rows


def write_manifest_csv(path: Path, selected: list[SelectedGame]) -> None:
    rows = build_csv_rows(selected)
    fieldnames = [
        "filename",
        "date",
        "white",
        "black",
        "result",
        "site",
        "meta_filename",
        "source_url",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> int:
    archive_url = args.archive_url.rstrip("/") + "/"
    out_dir = Path(args.out_dir)
    games_dir = out_dir / "games"
    meta_dir = out_dir / "meta"
    raw_dir = out_dir / "source"

    html = fetch_text(archive_url, args.timeout)
    links = extract_links(html, archive_url)
    link_set = set(links)
    pgn_urls = sorted((url for url in links if url.endswith(".pgn")), key=filename_from_url)
    if not pgn_urls:
        raise RuntimeError(f"no PGN links found at {archive_url}")
    if args.scan_mode == "filename":
        pgn_urls_to_scan = [
            url for url in pgn_urls if args.filename_match.casefold() in filename_from_url(url).casefold()
        ]
        if not pgn_urls_to_scan:
            raise RuntimeError(
                f"no PGN filenames matched {args.filename_match!r}; "
                "rerun with --scan-mode all to scan every archive PGN"
            )
    else:
        pgn_urls_to_scan = pgn_urls

    selected: list[SelectedGame] = []
    pgn_payloads: dict[str, bytes] = {}

    for url in pgn_urls_to_scan:
        payload = fetch_bytes(url, args.timeout)
        text = payload.decode("utf-8", errors="replace")
        tags = parse_pgn_tags(text)
        white = tags.get("White", "")
        black = tags.get("Black", "")
        if not (player_matches(white, args.engine) or player_matches(black, args.engine)):
            continue

        filename = filename_from_url(url)
        meta_filename, meta_url = find_matching_meta(url, link_set)
        selected.append(SelectedGame(filename, url, meta_filename, meta_url, tags))
        pgn_payloads[filename] = payload

    selected.sort(key=lambda game: sort_key(game.filename))
    if not selected:
        raise RuntimeError(f"no PGNs matched engine {args.engine!r}")

    out_dir.mkdir(parents=True, exist_ok=True)
    write_text(raw_dir / "archive-index.html", html)

    results_url = urljoin(archive_url, "tournament-results.json")
    tournament_results: dict[str, object] = {}
    try:
        results_payload = fetch_bytes(results_url, args.timeout)
        write_bytes(raw_dir / "tournament-results.json", results_payload)
        tournament_results = json.loads(results_payload.decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        print(f"warning: could not fetch/parse tournament results: {exc}", file=sys.stderr)

    meta_count = 0
    combined_chunks: list[str] = []
    for game in selected:
        pgn_data = pgn_payloads[game.filename]
        write_bytes(games_dir / game.filename, pgn_data)
        combined_chunks.append(pgn_data.decode("utf-8", errors="replace").strip())

        if game.meta_url and game.meta_filename:
            meta_payload = fetch_bytes(game.meta_url, args.timeout)
            write_bytes(meta_dir / game.meta_filename, meta_payload)
            meta_count += 1

    combined_name = f"{archive_slug(archive_url)}_{args.engine.lower().replace(' ', '_')}_all.pgn"
    write_text(out_dir / combined_name, "\n\n".join(combined_chunks) + "\n")

    manifest = {
        "source": "CCRL Live PGN archive",
        "archive_url": archive_url,
        "tournament_results_url": results_url,
        "engine": args.engine,
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "archive_pgn_count": len(pgn_urls),
        "selected_pgn_count": len(selected),
        "selected_meta_count": meta_count,
        "crosstable_ember_games": crosstable_ember_games(tournament_results, args.engine),
        "crosstable_ember_entries": crosstable_ember_entries(tournament_results, args.engine),
        "combined_pgn": combined_name,
        "games": [
            {
                "filename": game.filename,
                "url": game.url,
                "meta_filename": game.meta_filename,
                "meta_url": game.meta_url,
                "tags": game.tags,
            }
            for game in selected
        ],
    }
    write_text(out_dir / "manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    write_manifest_csv(out_dir / "manifest.csv", selected)

    print(f"archive PGNs found: {len(pgn_urls)}")
    print(f"archive PGNs scanned: {len(pgn_urls_to_scan)}")
    print(f"Ember PGNs downloaded: {len(selected)}")
    print(f"metadata files downloaded: {meta_count}")
    print(f"output: {out_dir}")
    if manifest["crosstable_ember_games"] is not None:
        print(f"crosstable Ember games: {manifest['crosstable_ember_games']}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-url", default=DEFAULT_ARCHIVE_URL)
    parser.add_argument("--engine", default=DEFAULT_ENGINE)
    parser.add_argument("--filename-match", default="ember_1.1.1")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--scan-mode", choices=["filename", "all"], default="filename")
    parser.add_argument("--timeout", type=int, default=30)
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
