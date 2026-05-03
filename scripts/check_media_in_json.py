#!/usr/bin/env python3
"""Check whether media paths in dataset entries exist on disk.

Expected dataset format: a JSON array of objects. Each object may include
`images`, `audios`, and `videos` fields containing path strings.

Usage:
    python scripts/check_media_paths.py data.json
"""

from __future__ import annotations

import argparse
import json
from tqdm import tqdm
from pathlib import Path
from typing import Any, Iterable

MEDIA_FIELDS = ("images", "audios", "videos")


def _iter_paths(value: Any) -> Iterable[str]:
    """Yield path strings from a field value.

    Supports:
    - list of strings
    - a single string
    - list of dicts containing one of: path, file, filepath, file_path
    """
    if value is None:
        return

    if isinstance(value, str):
        yield value
        return

    if isinstance(value, list):
        for item in value:
            if isinstance(item, str):
                yield item
            elif isinstance(item, dict):
                for key in ("path", "file", "filepath", "file_path"):
                    candidate = item.get(key)
                    if isinstance(candidate, str):
                        yield candidate
                        break


def check_file(dataset_path: Path) -> list[tuple[int, str, str]]:
    with dataset_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of objects.")

    missing: list[tuple[int, str, str]] = []

    for i, entry in tqdm(enumerate(data), desc="checking"):
        if not isinstance(entry, dict):
            continue

        for field in MEDIA_FIELDS:
            for raw_path in _iter_paths(entry.get(field)):
                # Skip remote URLs because they are not local filesystem paths.
                if raw_path.startswith("http://") or raw_path.startswith("https://"):
                    continue

                # User guarantees local paths are absolute; treat non-absolute as invalid.
                resolved = Path(raw_path)
                if (not resolved.is_absolute()) or (not resolved.exists()):
                    missing.append((i, field, raw_path))

    return missing


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check whether image/audio/video paths in a JSON dataset exist."
    )
    parser.add_argument("json_file", type=Path, help="Path to dataset JSON file")
    args = parser.parse_args()

    missing = check_file(args.json_file)

    if not missing:
        print("check pass")
        return

    print(f"Found {len(missing)} missing file path(s):")
    for idx, field, raw_path in missing:
        print(f"- entry[{idx}] field={field} path={raw_path}")


if __name__ == "__main__":
    main()
