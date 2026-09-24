"""Validação e expansão de modelos, prompts e recortes compartilhadas pelos executores."""

from __future__ import annotations

from pathlib import Path

from ..project import load_named
from ..prompts import PromptSet

PRICING_KEYS = ("input_usd_per_million", "output_usd_per_million", "cached_usd_per_million")


def freeze_models(root: Path, items: list) -> list[dict]:
    models = []
    for item in items or []:
        model = load_named(root, "models", item)
        if not model.get("id"):
            raise ValueError(f"Modelo sem id: {item}")
        pricing = model.get("pricing") or {}
        if any(key not in pricing for key in PRICING_KEYS):
            raise ValueError(f"{model['id']}: pricing requer {', '.join(PRICING_KEYS)}")
        models.append({"name": model.get("name", model["id"]), "id": str(model["id"]),
                       "extra_body": model.get("extra_body") or {}, "pricing": pricing})
    if not models or len({m["id"] for m in models}) != len(models):
        raise ValueError("experiment.models deve conter modelos com IDs distintos")
    return models


def freeze_prompt_set(root: Path, item: str | dict) -> dict:
    return PromptSet.from_dict(load_named(root, "prompts", item)).to_dict()


def prompt_set(experiment: dict) -> PromptSet:
    return PromptSet.from_dict(experiment["prompt_set"])


def selected(values: list, choice: list | None, name: str, key=lambda value: value) -> list:
    """Recorte ``select.<name>`` do plano congelado (``None`` = tudo)."""
    if choice is None:
        return list(values)
    wanted = list(choice)
    known = [key(value) for value in values]
    unknown = [item for item in wanted if item not in known]
    if not wanted or unknown or len(set(wanted)) != len(wanted):
        raise ValueError(f"select.{name} deve ser um subconjunto sem repetições de {known}")
    return [value for value in values if key(value) in wanted]


def select_models(models: list[dict], choice: list | None) -> list[dict]:
    if choice is None:
        return list(models)
    # Aceita tanto o ID da DeepInfra quanto o nome do arquivo em conf/models/.
    ids = [next((m["id"] for m in models if item in (m["id"], m.get("name"))), item)
           for item in choice]
    return selected(models, ids, "models", key=lambda model: model["id"])
