"""Conversão de runs criados antes de experiment.yaml (config.json antigo).

A conversão só é aceita se reproduzir exatamente os hashes registrados no run
(prompts, na classificação; chaves de todas as requisições, na viabilidade).
Os arquivos antigos não são alterados.
"""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath, PureWindowsPath

import yaml

from .io import now_utc, read_csv
from .project import FROZEN_FORMAT, load_named
from .prompts import PromptSet

PROMPT_SET = "veracidade_v1"


def _legacy_prompt_set(root: Path, introductions: dict | None, hashes: dict) -> dict:
    prompts = PromptSet.from_dict(load_named(root, "prompts", PROMPT_SET))
    if introductions is not None and introductions != prompts.introductions:
        raise ValueError(f"Introduções do run diferem de conf/prompts/{PROMPT_SET}.yaml")
    for prompt_id, digest in hashes.items():
        if prompts.fingerprint(prompt_id) != digest:
            raise ValueError(f"Prompt {prompt_id} do run difere de conf/prompts/{PROMPT_SET}.yaml")
    return prompts.to_dict()


def _folder_name(path: str) -> str:
    return (PureWindowsPath(path) if "\\" in path else PurePosixPath(path)).name


def _pilot(out: Path, root: Path, config: dict) -> dict:
    settings = config["request_settings"]
    if settings.get("n") != 1 or settings.get("stream") is not False:
        raise ValueError("Run legado com n/stream inesperados")
    responses = read_csv(out / "responses.csv")
    prompts = sorted({row["prompt_id"] for row in responses})
    if not prompts:
        resolved = out / "resolved_config.yaml"
        prompts = (yaml.safe_load(resolved.read_text(encoding="utf-8"))["experiment"]["prompts"]
                   if resolved.exists() else ["A"])
    manifest = read_csv(out / "manifest.csv")
    texts = PurePosixPath(manifest[0]["relative_path"]).parts[0] if manifest else "full_texts"
    return {
        "format": FROZEN_FORMAT, "experiment_name": "fakebr_multimodel",
        "runner": "multimodel", "created_at": config["created_at"],
        "converted_from": "config.json", "converted_at": now_utc(),
        "experiment": {
            "runner": "multimodel", "description": "Run legado convertido",
            "dataset": {"name": "fakebr", "loader": "fakebr_pairs",
                        "dir": _folder_name(config["dataset_root"]), "texts": texts,
                        "revision": config.get("dataset_git_revision"), "fingerprint": None},
            "seed": int(config["seed"]),
            "dev_pairs": int(config["dev_pairs"]), "eval_pairs": int(config["eval_pairs"]),
            "prompt_set": _legacy_prompt_set(root, config.get("prompt_introductions"),
                                             settings["prompt_sha256"]),
            "prompts": prompts,
            "models": settings["models"],
            "conditions": [{"temperature": float(c["temperature"]),
                            "repetitions": int(c["repetitions"])} for c in settings["conditions"]],
            "top_p": float(settings["top_p"]), "max_tokens": int(settings["max_tokens"]),
            "bootstrap_repeats": 500,
        },
        "api": {"base_url": settings["base_url"],
                "enable_prompt_cache_key": bool(settings["enable_prompt_cache_key"])},
    }


def _viability(out: Path, root: Path, config: dict) -> dict:
    from .experiments.viabilidade import controls_fingerprint, plans

    settings = config["settings"]
    experiment = settings["experiment"]
    frozen = {
        "format": FROZEN_FORMAT, "experiment_name": "viabilidade", "runner": "viabilidade",
        "created_at": config.get("created_at", ""),
        "converted_from": "config.json", "converted_at": now_utc(),
        "experiment": {
            "runner": "viabilidade", "description": "Run legado convertido",
            "prompt_set": _legacy_prompt_set(root, None, {}), "prompt": "A",
            "models": experiment["models"],
            "top_p": float(experiment["top_p"]), "max_tokens": int(experiment["max_tokens"]),
            "include_logprobs_probe": bool(settings["viability"]["include_logprobs_probe"]),
            "controls_sha256": controls_fingerprint(),
        },
        "api": {"base_url": settings["api"]["base_url"],
                "enable_prompt_cache_key": bool(settings["api"]["enable_prompt_cache_key"])},
    }
    if config.get("control_sha256") != frozen["experiment"]["controls_sha256"]:
        raise ValueError("Textos de controle do run diferem dos atuais")
    keys = [p["key"] for p in plans(frozen["experiment"], frozen["api"])]
    if keys != config["request_sha256"]:
        raise ValueError("As requisições reconstruídas não batem com as registradas no run")
    return frozen


def convert(out: Path, root: Path) -> dict | None:
    path = out / "config.json"
    if not path.exists():
        return None
    config = json.loads(path.read_text(encoding="utf-8"))
    try:
        if "request_settings" in config:
            return _pilot(out, root, config)
        if "payloads_sha256" in config and "settings" in config:
            return _viability(out, root, config)
    except (KeyError, ValueError) as exc:
        raise ValueError(f"Não foi possível converter o run legado {out.name}: {exc}") from exc
    return None
