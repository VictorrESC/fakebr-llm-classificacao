"""Cache local de respostas válidas, indexado pelo SHA-256 da requisição.

Arquivos pickle devem vir apenas deste projeto; nunca carregue um cache recebido
de terceiros, pois desserializar pickle pode executar código arbitrário.
"""

from __future__ import annotations

import os
import pickle
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path


HASH = re.compile(r"[0-9a-f]{64}\Z")
VERSION = 1


@dataclass(frozen=True)
class CacheSpec:
    """Como cada experimento guarda respostas: subpasta de cache/ e status aceitos."""

    subdir: str
    statuses: frozenset[str] = frozenset({"ok"})
    require_request_id: bool = False


class ResponseCache:
    def __init__(self, root: Path, statuses: frozenset[str] = frozenset({"ok"}),
                 require_request_id: bool = False):
        self.root = Path(root).expanduser().resolve()
        self.statuses = statuses
        self.require_request_id = require_request_id

    @classmethod
    def from_spec(cls, cache_root: Path, spec: CacheSpec) -> "ResponseCache":
        return cls(Path(cache_root) / spec.subdir, spec.statuses, spec.require_request_id)

    def path(self, key: str) -> Path:
        if not HASH.fullmatch(key):
            raise ValueError("request_sha256 inválido")
        return self.root / key[:2] / f"{key}.pkl"

    def get(self, key: str) -> dict | None:
        path = self.path(key)
        if not path.exists():
            return None
        with path.open("rb") as handle:
            stored = pickle.load(handle)
        if (not isinstance(stored, dict) or stored.get("version") != VERSION
                or stored.get("request_sha256") != key
                or not isinstance(stored.get("response"), dict)
                or stored["response"].get("status") not in self.statuses
                or (self.require_request_id and not stored["response"].get("request_id"))):
            raise ValueError(f"Cache inválido: {path}")
        return stored["response"]

    def put(self, row: dict) -> bool:
        if (row.get("status") not in self.statuses
                or (self.require_request_id and not row.get("request_id"))):
            return False
        key = row["request_sha256"]
        path = self.path(key)
        if path.exists():
            existing = self.get(key)
            if existing.get("raw_response") != row.get("raw_response"):
                raise ValueError(f"Resposta conflitante no cache: {key}")
            return False
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=".pending-", suffix=".pkl", dir=path.parent)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                pickle.dump({"version": VERSION, "request_sha256": key, "response": row},
                            handle, protocol=pickle.HIGHEST_PROTOCOL)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return True
