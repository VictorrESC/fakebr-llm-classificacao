"""Livro-caixa central dos custos dos experimentos DeepInfra.

Os executores gravam cada tentativa neste arquivo central, além do ``costs.csv``
local. ``pixi run costs`` também recupera linhas de pastas existentes, elimina
duplicatas e imprime um resumo acumulado.
"""

from __future__ import annotations

import csv
import os
from collections import defaultdict
from collections.abc import Callable
from decimal import Decimal, InvalidOperation
from pathlib import Path

from .io import canonical_hash, now_utc, read_csv


CENTRAL_FIELDS = [
    "ledger_id", "registered_at_utc", "source_type", "experiment_dir",
    "timestamp_utc", "stage", "probe", "control", "model", "temperature",
    "prompt_id", "repetition", "document_id", "request_sha256", "attempt",
    "status", "http_status", "prompt_tokens", "cached_tokens",
    "completion_tokens", "provider_cost_usd", "calculated_cost_usd",
    "accounted_cost_usd", "cost_source", "request_id", "error_type",
]


def _ledger_id(experiment_dir: Path, row: dict) -> str:
    # IDs da DeepInfra identificam cobranças reais mesmo se a pasta for copiada.
    # Erros sem request_id usam o nome do run (não o caminho absoluto, que muda
    # ao mover o projeto) e a data para distinguir tentativas.
    request_id = (row.get("request_id") or "").strip()
    if request_id:
        identity = {
            "request_id": request_id,
            "attempt": str(row.get("attempt", "")),
            "model": str(row.get("model", "")),
        }
    else:
        identity = {
            "experiment_dir": Path(experiment_dir).name,
            "timestamp_utc": str(row.get("timestamp_utc", "")),
            "request_sha256": str(row.get("request_sha256", "")),
            "attempt": str(row.get("attempt", "")),
            "status": str(row.get("status", "")),
            "error_type": str(row.get("error_type", "")),
        }
    return canonical_hash(identity)


def _money(value: object) -> Decimal:
    try:
        return Decimal(str(value)) if value not in (None, "") else Decimal("0")
    except InvalidOperation:
        return Decimal("0")


def accounted_cost(row: dict) -> str:
    """Custo do provedor quando informado; senão a estimativa local; senão vazio."""
    provider = row.get("provider_cost_usd", "")
    calculated = row.get("calculated_cost_usd", "")
    if provider not in (None, ""):
        return str(provider)
    if calculated not in (None, ""):
        return str(calculated)
    return ""


class CentralCostLedger:
    """Acrescenta custos a um CSV central sem duplicar tentativas."""

    def __init__(self, path: Path, experiment_dir: Path, source_type: str):
        self.path = Path(path).expanduser().resolve()
        self.experiment_dir = Path(experiment_dir).expanduser().resolve()
        self.source_type = source_type
        self.path.parent.mkdir(parents=True, exist_ok=True)
        existing = read_csv(self.path)
        self.ids = {row.get("ledger_id", "") for row in existing}
        if existing and list(existing[0]) != CENTRAL_FIELDS:
            raise RuntimeError(
                f"Cabeçalho incompatível no livro-caixa central: {self.path}"
            )

    def add(self, row: dict) -> bool:
        identifier = _ledger_id(self.experiment_dir, row)
        if identifier in self.ids:
            return False
        normalized = {field: row.get(field, "") for field in CENTRAL_FIELDS}
        normalized.update({
            "ledger_id": identifier,
            "registered_at_utc": now_utc(),
            "source_type": self.source_type,
            "experiment_dir": self.experiment_dir.name,
            "accounted_cost_usd": accounted_cost(row),
        })
        with self.path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=CENTRAL_FIELDS)
            if handle.tell() == 0:
                writer.writeheader()
            writer.writerow(normalized)
            handle.flush()
            os.fsync(handle.fileno())
        self.ids.add(identifier)
        return True

    def add_many(self, rows: list[dict]) -> int:
        return sum(self.add(row) for row in rows)


def infer_source_type(rows: list[dict]) -> str:
    """Tipo de origem para pastas sem experiment.yaml (runs muito antigos)."""
    if rows and "probe" in rows[0]:
        return "viability"
    if rows and "stage" in rows[0]:
        return "pilot"
    return "unknown"


def sync_project(runs_dir: Path, ledger_path: Path,
                 source_of: Callable[[Path, list[dict]], str] | None = None) -> int:
    added = 0
    for cost_file in sorted(Path(runs_dir).glob("*/costs.csv")):
        rows = read_csv(cost_file)
        if not rows:
            continue
        source = source_of(cost_file.parent, rows) if source_of else infer_source_type(rows)
        added += CentralCostLedger(ledger_path, cost_file.parent, source).add_many(rows)
    return added


def print_report(ledger_path: Path) -> None:
    rows = read_csv(ledger_path)
    total = sum((_money(row.get("accounted_cost_usd")) for row in rows), Decimal("0"))
    unknown = sum(row.get("accounted_cost_usd", "") == "" for row in rows)
    by_experiment: dict[str, Decimal] = defaultdict(Decimal)
    by_model: dict[str, Decimal] = defaultdict(Decimal)
    for row in rows:
        value = _money(row.get("accounted_cost_usd"))
        by_experiment[row.get("experiment_dir", "")] += value
        by_model[row.get("model", "")] += value

    print(f"Livro-caixa: {Path(ledger_path).resolve()}")
    print(f"Tentativas registradas: {len(rows)}")
    print(f"Custo total contabilizado: US$ {total:.8f}")
    print(f"Tentativas com custo desconhecido: {unknown}")
    if by_experiment:
        print("\nPor experimento:")
        for name, value in sorted(by_experiment.items()):
            print(f"  {name}: US$ {value:.8f}")
    if by_model:
        print("\nPor modelo:")
        for name, value in sorted(by_model.items()):
            print(f"  {name}: US$ {value:.8f}")
