"""Entrada única das tarefas Pixi. Experimentos vêm de conf/experiment/*.yaml."""

from __future__ import annotations

import argparse
import sys

from . import costs, data, project
from .io import read_csv

CREATED_FILES = (project.FROZEN_FILE, "provenance.json", "executions.jsonl")


def _parser(experiments: list[str]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fakebr", description="Executa e analisa experimentos")
    sub = parser.add_subparsers(dest="command", required=True)
    for command, help_text in (("experiment", "Executa uma etapa (pode chamar a API)"),
                               ("report", "Gera relatório de um run existente (sem API)")):
        item = sub.add_parser(command, help=help_text)
        item.add_argument("name", choices=experiments, help="Arquivo de conf/experiment/")
        item.add_argument("--run", required=True, help="Nome da pasta sob runs/")
        item.add_argument("--stage", help="Etapa; veja 'pixi run status'")
        item.add_argument("overrides", nargs="*", help="Overrides Hydra, ex.: api.concurrency=8")
    item = sub.add_parser("cache-import", help="Copia respostas de um run para cache/")
    item.add_argument("--run", required=True)
    item = sub.add_parser("costs", help="Sincroniza e resume o livro-caixa central")
    item.add_argument("--no-sync", action="store_true",
                      help="Apenas mostra o CSV central, sem procurar costs.csv existentes")
    item = sub.add_parser("fetch-data", help="Baixa um corpus na revisão fixada em conf/")
    item.add_argument("--dataset", default="fakebr")
    sub.add_parser("status", help="Lista experimentos disponíveis e runs existentes")
    for name, command in sub.choices.items():
        if name not in ("experiment", "report"):
            command.add_argument("overrides", nargs="*", help="Overrides, ex.: paths.runs=outra/pasta")
    return parser


def _base(args) -> tuple:
    root = project.project_root()
    cfg = project.compose(root, None, args.overrides)
    return root, cfg, project.Paths.from_config(root, cfg["paths"])


def _stage(module, command: str, stage: str | None) -> str:
    if command == "experiment":
        options, default = module.STAGES, (module.STAGES[0] if len(module.STAGES) == 1 else None)
    else:
        options, default = module.REPORT_STAGES, module.DEFAULT_REPORT_STAGE
    stage = stage or default
    if stage not in options:
        raise ValueError(f"Informe --stage {' | '.join(options)}")
    return stage


def _discard_new_run(out) -> None:
    """Remove um run recém-criado cuja primeira etapa falhou sem produzir nada."""
    if not out.is_dir() or any(p.name not in CREATED_FILES for p in out.iterdir()):
        return
    for path in out.iterdir():
        path.unlink()
    out.rmdir()


def execute(command: str, name: str, run_name: str, stage: str | None = None,
            overrides: list[str] | tuple[str, ...] = ()):
    """Mesmo efeito de ``pixi run <command> <name> --run <run_name>``; para notebooks.

    ``command`` é "experiment" ou "report". No relatório, devolve um dicionário
    com ``report_dir``, ``tables`` e ``figures`` (caminhos dos arquivos gerados).
    """
    if command not in ("experiment", "report"):
        raise ValueError("command deve ser 'experiment' ou 'report'")
    overrides = list(overrides)
    root = project.project_root()
    module = project.runner(project.compose(root, name, overrides)["experiment"]["runner"])
    stage = _stage(module, command, stage)
    run, module, new = project.open_run(
        name, run_name, overrides, create_stage=stage if command == "experiment" else None)
    project.record_execution(run, root, command, stage, overrides)
    try:
        return (module.run if command == "experiment" else module.report)(run, stage)
    except BaseException:
        if new:
            _discard_new_run(run.out)
        raise


def _experiment(args) -> None:
    execute(args.command, args.name, args.run, args.stage, args.overrides)


def _cache_import(args) -> None:
    run, module, _ = project.open_run(None, args.run, args.overrides)
    rows = read_csv(run.out / module.RESULTS_FILE)
    if not rows:
        raise ValueError("Nenhuma resposta encontrada neste run")
    cache = run.cache(module.CACHE)
    count = sum(cache.put(row) for row in rows)
    print(f"{count} respostas importadas; {len(rows)} examinadas")


def _source_of(paths: project.Paths):
    root = paths.root

    def source(run_dir, rows) -> str:
        try:
            frozen = project.read_frozen(run_dir, root)
        except (ValueError, KeyError):
            frozen = None
        if frozen is None:
            return costs.infer_source_type(rows)
        return project.runner(frozen["runner"]).LEDGER_SOURCE
    return source


def _costs(args) -> None:
    _, _, paths = _base(args)
    if not args.no_sync:
        added = costs.sync_project(paths.runs, paths.ledger, _source_of(paths))
        print(f"Novas tentativas importadas: {added}")
    costs.print_report(paths.ledger)


def _fetch(args) -> None:
    _, cfg, paths = _base(args)
    if args.dataset not in cfg["datasets"]:
        raise ValueError(f"Corpus desconhecido: {args.dataset}")
    source = cfg["datasets"][args.dataset]
    destination = paths.data / source["dir"]
    if data.fetch(source["repository"], source["revision"], destination):
        print(f"Corpus em {destination} (revisão {source['revision']}).")
    else:
        print(f"{destination} já existe; nada a baixar. O 'prepare' confere a versão dos textos.")


def _status(args) -> None:
    root, _, paths = _base(args)
    print("Experimentos (conf/experiment/):")
    for name in project.experiment_names(root):
        experiment = project.compose(root, name, [])["experiment"]
        module = project.runner(experiment["runner"])
        print(f"  {name:<24} etapas: {', '.join(module.STAGES):<20} {experiment.get('description', '')}")
    runs_dir = paths.runs
    print(f"\nRuns ({runs_dir}):")
    for folder in sorted(p for p in runs_dir.glob("*") if p.is_dir() and p.name != "hydra"):
        try:
            frozen = project.read_frozen(folder, root)
        except ValueError as exc:
            print(f"  {folder.name:<40} ERRO: {exc}")
            continue
        print(f"  {folder.name:<40} {frozen['experiment_name'] if frozen else '(sem experiment.yaml)'}")


def run_cli(argv: list[str] | None = None) -> None:
    root = project.project_root()
    args = _parser(project.experiment_names(root)).parse_args(argv)
    {"experiment": _experiment, "report": _experiment, "cache-import": _cache_import,
     "costs": _costs, "fetch-data": _fetch, "status": _status}[args.command](args)


def main() -> None:
    from hydra.errors import HydraException

    try:
        run_cli()
    except (ValueError, RuntimeError, OSError, HydraException) as exc:
        print(f"Erro: {exc}", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
