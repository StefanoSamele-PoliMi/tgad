"""Small deterministic utilities used throughout the repository."""

from __future__ import annotations

import contextlib
import csv
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence


_NATURAL_RE = re.compile(r"(\d+)")


def natural_key(value: str) -> tuple[Any, ...]:
    """Return a key that sorts embedded integers numerically."""

    return tuple(int(part) if part.isdigit() else part.lower() for part in _NATURAL_RE.split(value))


def stable_json_dumps(value: Any) -> str:
    """Serialize JSON deterministically for hashing and manifests."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(stable_json_dumps(value).encode("utf-8")).hexdigest()


def file_identity(path: str | Path) -> dict[str, Any]:
    """Return a cheap identity used in prediction cache keys."""

    p = Path(path).expanduser().resolve()
    stat = p.stat()
    return {
        "path": str(p),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def atomic_write_text(path: str | Path, text: str) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=target.parent, delete=False) as handle:
        handle.write(text)
        temp_path = Path(handle.name)
    os.replace(temp_path, target)


def write_json(path: str | Path, value: Any, *, indent: int = 2) -> None:
    atomic_write_text(path, json.dumps(value, indent=indent, ensure_ascii=False, sort_keys=True) + "\n")


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    if fieldnames is None:
        keys: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    keys.append(key)
        fieldnames = keys
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=target.parent, delete=False) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        temp_path = Path(handle.name)
    os.replace(temp_path, target)


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


@contextlib.contextmanager
def pushd(path: str | Path) -> Iterator[None]:
    """Temporarily change the process working directory."""

    previous = Path.cwd()
    os.chdir(Path(path))
    try:
        yield
    finally:
        os.chdir(previous)


def chunked(values: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    if size <= 0:
        raise ValueError("size must be positive")
    for start in range(0, len(values), size):
        yield values[start : start + size]
