#!/usr/bin/env python3
"""Rewrite training_media_dir placeholder in a JSON file and overwrite it."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

PLACEHOLDER = "{training_media_dir}/"
MEDIA_FIELDS = ("images", "audios", "videos")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild data from _data and rewrite media placeholders in all JSONs."
    )
    parser.add_argument(
        "target_prefix",
        type=Path,
        help="Absolute or relative path to extracted media root.",
    )
    return parser.parse_args()


def _rewrite_media_field(value, abs_target_prefix: str):
    if isinstance(value, str):
        if value.startswith(PLACEHOLDER):
            return value.format(training_media_dir=abs_target_prefix.rstrip('/'))
        return value

    if isinstance(value, list):
        return [_rewrite_media_field(item, abs_target_prefix) for item in value]

    return value


def _resolve_target_prefix(target_prefix: Path) -> str:
    target_path = target_prefix.resolve()
    if not target_path.exists():
        raise FileNotFoundError(f"target prefix does not exist: {target_path}")
    return str(target_path)


def _rebuild_data_dir(ms_swift_dir: Path) -> Path:
    source_dir = ms_swift_dir / "_data"
    dest_dir = ms_swift_dir / "data"

    if not source_dir.exists() or not source_dir.is_dir():
        raise FileNotFoundError(f"source _data directory does not exist: {source_dir}")

    if dest_dir.exists():
        shutil.rmtree(dest_dir)

    shutil.copytree(source_dir, dest_dir)
    return dest_dir


def _rewrite_json_file(json_path: Path, abs_target_prefix: str) -> None:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"input JSON must be a list of objects: {json_path}")

    for row in data:
        if not isinstance(row, dict):
            continue

        for field in MEDIA_FIELDS:
            if field in row:
                row[field] = _rewrite_media_field(row[field], abs_target_prefix)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
        f.write("\n")


def main() -> int:
    args = parse_args()

    ms_swift_dir = Path(__file__).resolve().parent
    abs_target_prefix = _resolve_target_prefix(args.target_prefix)
    data_dir = _rebuild_data_dir(ms_swift_dir)

    json_files = sorted(data_dir.rglob("*.json"))
    for json_file in json_files:
        _rewrite_json_file(json_file, abs_target_prefix)

    print(f"[INFO] target prefix: {abs_target_prefix}")
    print(f"[INFO] rebuilt data dir: {data_dir}")
    print(f"[INFO] processed json files: {len(json_files)}")
    return 0


if __name__ == "__main__":
    main()
