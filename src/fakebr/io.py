"""Leitura e escrita de CSV/JSON e hashes canônicos usados por todos os experimentos."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


def sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonical_hash(value: object) -> str:
    # Mudar esta serialização invalida o cache e as chaves de todos os runs.
    return sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                             separators=(",", ":")))


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def atomic_json(path: Path, value: object) -> None:
    _atomic_write(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def atomic_text(path: Path, text: str) -> None:
    _atomic_write(path, text)


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _header(path: Path) -> list[str] | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return next(csv.reader(handle), None)


def append_csv(path: Path, row: dict, fields: list[str]) -> None:
    """Acrescenta uma linha e força a gravação em disco.

    Um arquivo existente mantém o próprio cabeçalho; colunas novas não são
    aceitas, para não deslocar silenciosamente os valores de runs antigos.
    """
    header = _header(path)
    if header is not None and set(fields) - set(header):
        raise RuntimeError(f"Cabeçalho de {path} não contém {sorted(set(fields) - set(header))}")
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=header or fields, extrasaction="ignore")
        if header is None:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)
