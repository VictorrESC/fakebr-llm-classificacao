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
    for command, help_text, run_help in (
            ("experiment", "Executa uma etapa (pode chamar a API)",
             "Nome da pasta sob runs/ (obrigatório: etapas podem gerar custos)"),
            ("report", "Gera relatório de um run existente (sem API)",
             "Nome da pasta sob runs/ (padrão: o último run usado)")):
        item = sub.add_parser(command, help=help_text)
        item.add_argument("name", nargs="?", metavar="experimento",
                          help=f"Arquivo de conf/experiment/ ({', '.join(experiments)}); "
                               "só é preciso ao criar um run, nos demais casos vem do run")
        item.add_argument("--run", help=run_help)
        item.add_argument("--stage", help="Etapa; no relatório, padrão = a mais avançada com respostas")
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


def _stage(module, command: str, stage: str | None, run=None) -> str:
    if command == "experiment":
        options, default = module.STAGES, (module.STAGES[0] if len(module.STAGES) == 1 else None)
    else:
        # A etapa mais avançada com respostas, quando o executor sabe dizer.
        pick = getattr(module, "default_report_stage", None)
        options = module.REPORT_STAGES
        default = pick(run) if pick and run is not None else module.DEFAULT_REPORT_STAGE
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


def execute(command: str, name: str | None = None, run_name: str | None = None,
            stage: str | None = None, overrides: list[str] | tuple[str, ...] = ()):
    """Mesmo efeito de ``pixi run <command> [<name>] --run <run_name>``; para notebooks.

    ``command`` é "experiment" ou "report". ``name`` (experimento) só é
    necessário ao criar um run; nos demais casos vem do ``experiment.yaml`` do
    run. No relatório, ``run_name`` e ``stage`` também são opcionais: o padrão é o
    último run usado e a etapa mais avançada com respostas. Devolve, no
    relatório, um dicionário com ``report_dir``, ``tables`` e ``figures``.
    """
    if command not in ("experiment", "report"):
        raise ValueError("command deve ser 'experiment' ou 'report'")
    overrides = list(overrides)
    root = project.project_root()
    paths = project.paths_for(root, overrides)
    if run_name is None:
        if command == "experiment":
            raise ValueError("Informe --run <nome>: etapas de experimento podem gerar custos.")
        run_name = project.current_run(paths)
        if run_name is None:
            raise ValueError("Nenhum run usado ainda; informe --run <nome> (veja 'pixi run status').")
    frozen = project.frozen_of(paths, root, run_name)
    if name is None:
        if frozen is None:
            raise ValueError(f"O run '{run_name}' não existe. Para criá-lo, informe o experimento: "
                             f"pixi run experiment <experimento> --run {run_name} --stage <primeira etapa>")
        name = frozen["experiment_name"]
    module = project.runner(frozen["runner"] if frozen
                            else project.compose(root, name, overrides)["experiment"]["runner"])
    if command == "experiment":
        stage = _stage(module, command, stage)
        run, module, new = project.open_run(name, run_name, overrides, create_stage=stage)
    else:
        run, module, new = project.open_run(name, run_name, overrides)
        stage = _stage(module, command, stage, run)
    print(f"Run: {run_name} (experimento {name}), etapa {stage}", flush=True)
    project.record_execution(run, root, command, stage, overrides)
    if not new:
        project.set_current_run(paths, run_name)
    try:
        result = (module.run if command == "experiment" else module.report)(run, stage)
    except BaseException:
        if new:
            _discard_new_run(run.out)
        raise
    project.set_current_run(paths, run_name)
    return result


def _experiment(args) -> None:
    name, overrides = args.name, list(args.overrides)
    if name is not None and "=" in name:  # sem experimento: o 1º argumento já é um override
        name, overrides = None, [name, *overrides]
    experiments = project.experiment_names(project.project_root())
    if name is not None and name not in experiments:
        raise ValueError(f"Experimento desconhecido: {name}. Disponíveis: {', '.join(experiments)}")
    execute(args.command, name, args.run, args.stage, overrides)


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
    current = project.current_run(paths)
    print(f"\nRuns ({runs_dir}); * = run atual, usado por 'pixi run report' sem --run:")
    for folder in sorted(p for p in runs_dir.glob("*") if p.is_dir() and p.name != "hydra"):
        mark = "*" if folder.name == current else " "
        try:
            frozen = project.read_frozen(folder, root)
        except ValueError as exc:
            print(f" {mark}{folder.name:<40} ERRO: {exc}")
            continue
        print(f" {mark}{folder.name:<40} {frozen['experiment_name'] if frozen else '(sem experiment.yaml)'}")


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
