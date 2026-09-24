"""Executores de experimentos.

Cada ``conf/experiment/<nome>.yaml`` indica ``experiment.runner``, o nome de um
módulo deste pacote. Um módulo executor define:

- ``STAGES``: etapas aceitas por ``pixi run experiment``;
- ``FIRST_STAGES``: etapas que podem criar um run novo;
- ``REPORT_STAGES`` e ``DEFAULT_REPORT_STAGE``: opções de ``pixi run report``;
- ``LEDGER_SOURCE``, ``RESULTS_FILE`` e ``CACHE``: livro-caixa e cache;
- ``freeze(cfg, root) -> dict``: expande e valida a seção ``experiment``;
- ``run(run, stage)``;
- ``report(run, stage) -> dict`` com ``report_dir``, ``tables`` e ``figures``
  (use ``plots.output_formats(run.report)`` para respeitar ``report.formats``).

Experimentos que só mudam modelos, prompts, temperaturas ou amostra reutilizam um
executor existente: basta um YAML novo em ``conf/experiment/``.
"""
