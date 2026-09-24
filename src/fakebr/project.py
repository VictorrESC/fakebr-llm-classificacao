"""Raiz do projeto, composição Hydra e configuração congelada de cada run.

Fluxo: a primeira etapa de um run compõe ``conf/config.yaml`` +
``conf/experiment/<nome>.yaml`` + overrides, expande modelos e prompts e grava
tudo em ``runs/<run>/experiment.yaml``. As etapas seguintes e os relatórios
leem apenas esse arquivo; do ``conf/`` atual vêm somente caminhos, opções de
ritmo da API e recortes (``select``). Assim, editar ``conf/`` para um novo
experimento nunca altera nem quebra runs antigos.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from types import ModuleType

import yaml

from .api import REQUEST_KEYS, RUNTIME_KEYS
from .cache import CacheSpec, ResponseCache
from .costs import CentralCostLedger
from .io import atomic_json, atomic_text, now_utc, sha256

FROZEN_FILE = "experiment.yaml"
FROZEN_FORMAT = 1
RESUME_OVERRIDES = tuple(f"api.{key}" for key in RUNTIME_KEYS) + ("select.", "paths.", "report.")
PACKAGES = ("hydra-core", "omegaconf", "openai", "numpy", "scikit-learn", "matplotlib")


def project_root() -> Path:
    # Instalação editável: src/fakebr/project.py -> raiz. FAKEBR_ROOT permite outra raiz.
    root = Path(os.environ.get("FAKEBR_ROOT") or Path(__file__).resolve().parents[2]).resolve()
    if not (root / "conf" / "config.yaml").is_file():
        raise FileNotFoundError(f"conf/config.yaml não encontrado em {root}")
    return root


def compose(root: Path, experiment: str | None, overrides: list[str]) -> dict:
    from hydra import compose as hydra_compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf

    if experiment is not None and experiment not in experiment_names(root):
        raise ValueError(f"Experimento desconhecido: {experiment}. "
                         f"Disponíveis: {', '.join(experiment_names(root))}")
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(root / "conf"), version_base="1.3"):
        cfg = hydra_compose(config_name="config", overrides=(
            [f"experiment={experiment}"] if experiment else []) + list(overrides))
    return OmegaConf.to_container(cfg, resolve=True)


def experiment_names(root: Path) -> list[str]:
    return sorted(path.stem for path in (root / "conf" / "experiment").glob("*.yaml"))


def runner(name: str) -> ModuleType:
    """Módulo em fakebr/experiments/<runner>.py; nenhum registro manual."""
    try:
        return import_module(f"fakebr.experiments.{name}")
    except ModuleNotFoundError as exc:
        if exc.name == f"fakebr.experiments.{name}":
            raise ValueError(f"Executor inexistente: src/fakebr/experiments/{name}.py") from exc
        raise


def load_yaml(path: Path) -> dict:
    # yaml.safe_load, não OmegaConf: "${...}" num prompt não vira interpolação.
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _literal(dumper: yaml.SafeDumper, value: str):
    style = "|" if "\n" in value else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", value, style=style)


class _Dumper(yaml.SafeDumper):
    pass


_Dumper.add_representer(str, _literal)


def dump_yaml(value: dict) -> str:
    return yaml.dump(value, Dumper=_Dumper, allow_unicode=True, sort_keys=False, width=100)


def load_named(root: Path, kind: str, item: str | dict) -> dict:
    """Expande um nome (``conf/<kind>/<nome>.yaml``) ou aceita um dicionário inline."""
    if isinstance(item, dict):
        return dict(item)
    path = root / "conf" / kind / f"{item}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"conf/{kind}/{item}.yaml não existe")
    return {"name": str(item), **load_yaml(path)}


@dataclass(frozen=True)
class Paths:
    root: Path
    data: Path
    runs: Path
    cache: Path
    ledger: Path

    @classmethod
    def from_config(cls, root: Path, paths: dict) -> "Paths":
        def resolve(value: str) -> Path:
            path = Path(value).expanduser()
            return (path if path.is_absolute() else root / path).resolve()
        return cls(root, *(resolve(paths[key]) for key in ("data", "runs", "cache", "ledger")))

    def run_dir(self, name: str) -> Path:
        out = (self.runs / name).resolve()
        if not name or out.parent != self.runs or (out.exists() and not out.is_dir()):
            raise ValueError("--run deve ser o nome de uma pasta em runs/")
        return out


@dataclass
class Run:
    name: str
    out: Path
    paths: Paths
    frozen: dict
    api: dict
    select: dict
    report: dict

    @property
    def experiment(self) -> dict:
        return self.frozen["experiment"]

    def cache(self, spec: CacheSpec) -> ResponseCache:
        return ResponseCache.from_spec(self.paths.cache, spec)

    def ledger(self, source_type: str) -> CentralCostLedger:
        return CentralCostLedger(self.paths.ledger, self.out, source_type)


def code_fingerprint(root: Path) -> str:
    files = sorted([*(root / "src" / "fakebr").rglob("*.py"), *(root / "conf").rglob("*.yaml"),
                    root / "pyproject.toml"])
    digest = [(path.relative_to(root).as_posix(),
               sha256(path.read_bytes().replace(b"\r\n", b"\n").decode("utf-8")))
              for path in files if path.is_file()]
    return sha256(json.dumps(digest))


def git_info(root: Path) -> dict | None:
    if not (root / ".git").exists():
        return None
    try:
        commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                                capture_output=True, text=True, check=False)
        status = subprocess.run(["git", "-C", str(root), "status", "--porcelain"],
                                capture_output=True, text=True, check=False)
    except FileNotFoundError:
        return None
    return {"commit": commit.stdout.strip() or None, "dirty": bool(status.stdout.strip())}


def provenance(root: Path) -> dict:
    def package(name: str) -> str | None:
        try:
            return version(name)
        except PackageNotFoundError:
            return None

    lock = root / "pixi.lock"
    return {
        "recorded_at": now_utc(), "python": sys.version, "platform": platform.platform(),
        "packages": {name: package(name) for name in PACKAGES},
        "code_sha256": code_fingerprint(root), "git": git_info(root),
        "pixi_lock_sha256": sha256(lock.read_text(encoding="utf-8")) if lock.exists() else None,
    }


def read_frozen(out: Path, root: Path) -> dict | None:
    path = out / FROZEN_FILE
    if path.exists():
        frozen = load_yaml(path)
        if frozen.get("format") != FROZEN_FORMAT:
            raise ValueError(f"Formato desconhecido em {path}")
        return frozen
    from .legacy import convert

    frozen = convert(out, root)
    if frozen is not None:
        write_frozen(out, frozen)
        print(f"Run legado convertido: {path} criado (arquivos antigos preservados).")
    return frozen


def write_frozen(out: Path, frozen: dict) -> None:
    header = ("# Configuração congelada deste run. Não edite: etapas seguintes e\n"
              "# relatórios usam exatamente estes valores.\n")
    atomic_text(out / FROZEN_FILE, header + dump_yaml(frozen))


def _override_key(override: str) -> str:
    return override.lstrip("~+").split("=", 1)[0].strip()


def check_resume_overrides(overrides: list[str]) -> None:
    for override in overrides:
        key = _override_key(override)
        if not any(key == allowed or (allowed.endswith(".") and key.startswith(allowed))
                   for allowed in RESUME_OVERRIDES):
            raise ValueError(
                f"'{key}' está congelado neste run. Ao retomar ou relatar, só é possível "
                f"alterar {', '.join(RESUME_OVERRIDES)}. Para mudar o plano, crie um run novo "
                "(respostas idênticas virão do cache).")


def record_execution(run: Run, root: Path, command: str, stage: str | None,
                     overrides: list[str]) -> None:
    line = {"timestamp_utc": now_utc(), "command": command, "stage": stage,
            "overrides": overrides, "code_sha256": code_fingerprint(root),
            "git": git_info(root)}
    with (run.out / "executions.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(line, ensure_ascii=False) + "\n")


def open_run(experiment: str | None, run_name: str, overrides: list[str], *,
             create_stage: str | None = None) -> tuple[Run, ModuleType, bool]:
    """Carrega (ou cria, na primeira etapa) o run; devolve (run, executor, criado).

    ``experiment=None`` aceita qualquer run existente (cache-import, custos).
    """
    root = project_root()
    base = compose(root, None, [o for o in overrides if _override_key(o).startswith("paths.")])
    paths = Paths.from_config(root, base["paths"])
    out = paths.run_dir(run_name)
    frozen = read_frozen(out, root) if out.is_dir() else None
    created = frozen is None

    if frozen is None:
        if experiment is None:
            raise FileNotFoundError(f"O run '{run_name}' não existe (ou não tem {FROZEN_FILE}).")
        cfg = compose(root, experiment, overrides)
        module = runner(cfg["experiment"]["runner"])
        if create_stage not in module.FIRST_STAGES:
            raise ValueError(f"O run '{run_name}' não existe. Para criá-lo, comece por "
                             f"--stage {' ou '.join(module.FIRST_STAGES)}.")
        if out.exists() and any(out.iterdir()):
            raise FileExistsError(f"{out} não está vazia e não tem {FROZEN_FILE}; use outro --run.")
        frozen = {
            "format": FROZEN_FORMAT, "experiment_name": experiment,
            "runner": cfg["experiment"]["runner"], "created_at": now_utc(),
            "experiment": module.freeze(cfg, root),
            "api": {key: cfg["api"][key] for key in REQUEST_KEYS},
        }
        out.mkdir(parents=True, exist_ok=True)
        write_frozen(out, frozen)
        atomic_json(out / "provenance.json", provenance(root))
        print(f"Configuração congelada: {out / FROZEN_FILE}")
    else:
        if experiment is not None and frozen["experiment_name"] != experiment:
            raise ValueError(f"O run '{run_name}' pertence ao experimento "
                             f"'{frozen['experiment_name']}', não a '{experiment}'.")
        check_resume_overrides(overrides)
        name = frozen["experiment_name"]
        cfg = compose(root, name if name in experiment_names(root) else None, overrides)
        module = runner(frozen["runner"])
        for key in REQUEST_KEYS:
            if key in cfg["api"] and cfg["api"][key] != frozen["api"][key]:
                print(f"Aviso: api.{key} no conf/ difere do valor congelado; usando o do run.")

    api = {**cfg["api"], **frozen["api"]}
    return (Run(run_name, out, paths, frozen, api, dict(cfg.get("select") or {}),
                dict(cfg.get("report") or {})), module, created)
