"""Figuras dos relatórios, salvas em PNG, PDF e/ou SVG.

Usa a API orientada a objetos do matplotlib (sem pyplot): funciona igual no
terminal e em notebooks, sem trocar o backend da sessão.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from pathlib import Path

from .prompts import INVALID

# Metadados de data removidos: o mesmo relatório gera os mesmos bytes.
FORMATS = {"png": {}, "pdf": {"metadata": {"CreationDate": None}},
           "svg": {"metadata": {"Date": None}}}
DPI = 170
MULTIMODEL_FIGURES = ("macro_f1", "per_class", "confusion", "stability", "cost_by_condition")
VIABILITY_FIGURES = ("viability_status", "viability_cost")


def output_formats(options: dict | None) -> tuple[str, ...]:
    """Formatos pedidos em ``report.formats`` (padrão: png)."""
    formats = [str(item).lower().lstrip(".") for item in ((options or {}).get("formats") or ["png"])]
    unknown = sorted(set(formats) - set(FORMATS))
    if not formats or unknown:
        raise ValueError(f"report.formats aceita {sorted(FORMATS)}; recebido {unknown or formats}")
    return tuple(dict.fromkeys(formats))


def _figure(**kwargs):
    from matplotlib.figure import Figure

    return Figure(**kwargs)


def _save(fig, folder: Path, stem: str, formats: tuple[str, ...]) -> list[Path]:
    paths = []
    for fmt in formats:
        path = folder / f"{stem}.{fmt}"
        fig.savefig(path, dpi=DPI, **FORMATS[fmt])
        paths.append(path)
    return paths


def _clear(folder: Path, stems: tuple[str, ...], patterns: tuple[str, ...] = ()) -> None:
    for fmt in FORMATS:
        for stem in stems:
            (folder / f"{stem}.{fmt}").unlink(missing_ok=True)
        for pattern in patterns:
            for stale in folder.glob(f"{pattern}.{fmt}"):
                stale.unlink()


def condition_name(model_id: str, temperature: float, repetition: int, prompt_id: str) -> str:
    return f"{model_id}|t={temperature:g}|r={repetition}|p={prompt_id}"


def condition_parts(name: str) -> tuple[str, float, int, str]:
    model_id, temp_part, rep_part, prompt_part = name.split("|")
    return model_id, float(temp_part[2:]), int(rep_part[2:]), prompt_part[2:]


def model_label(model_id: str) -> str:
    return model_id.rsplit("/", 1)[-1]


def _short(labels: tuple[str, ...]) -> list[str]:
    initials = [label[0] for label in labels]
    return initials if len(set(initials)) == len(initials) else list(labels)


def figures(report_dir: Path, metrics: dict[str, dict], repeat_rows: list[dict],
            cost_rows: list[dict], stage: str, labels: tuple[str, ...],
            formats: tuple[str, ...] = ("png",)) -> list[Path]:
    """Figuras do relatório multimodelo; devolve os arquivos gravados."""
    import matplotlib
    import numpy as np

    _clear(report_dir, MULTIMODEL_FIGURES, ("confusion_prompt_*",))
    saved: list[Path] = []

    grouped = defaultdict(list)
    model_order, temperature_order, prompt_order = [], [], []
    for name, score in metrics.items():
        model_id, temperature, repetition, prompt_id = condition_parts(name)
        grouped[(model_id, temperature, prompt_id)].append((repetition, score))
        if model_id not in model_order:
            model_order.append(model_id)
        if temperature not in temperature_order:
            temperature_order.append(temperature)
        if prompt_id not in prompt_order:
            prompt_order.append(prompt_id)
    temperature_order.sort()
    x = np.arange(len(temperature_order), dtype=float)
    series = [(model_id, prompt_id) for model_id in model_order
              for prompt_id in prompt_order
              if any((model_id, temperature, prompt_id) in grouped
                     for temperature in temperature_order)]

    def series_label(model_id: str, prompt_id: str, prefix: str = "") -> str:
        label = model_label(model_id)
        return label + (f" / {prefix}{prompt_id}" if len(prompt_order) > 1 else "")

    # Macro-F1: pontos são repetições; linha e barra de erro são média e DP.
    fig = _figure(figsize=(9, 5))
    ax = fig.subplots()
    offsets = np.linspace(-0.18, 0.18, max(len(series), 1))
    colors = matplotlib.rcParams["axes.prop_cycle"].by_key()["color"]
    for series_index, (offset, (model_id, prompt_id)) in enumerate(zip(offsets, series)):
        color = colors[series_index % len(colors)]
        means, deviations = [], []
        for temperature in temperature_order:
            values = [item[1]["macro_f1"]
                      for item in grouped.get((model_id, temperature, prompt_id), [])]
            means.append(statistics.mean(values) if values else np.nan)
            deviations.append(statistics.pstdev(values) if len(values) > 1 else 0.0)
            ax.scatter(np.full(len(values), x[temperature_order.index(temperature)] + offset),
                       values, s=24, alpha=.55, color=color)
        ax.errorbar(x + offset, means, yerr=deviations, marker="o", capsize=4,
                    linewidth=1.8, label=series_label(model_id, prompt_id, "prompt "),
                    color=color)
    ax.set_xticks(x, [f"{temperature:g}" for temperature in temperature_order])
    ax.set_ylim(0, 1)
    ax.set_xlabel("Temperatura")
    ax.set_ylabel("Macro-F1")
    ax.set_title(f"Macro-F1 por modelo e temperatura — {stage}")
    ax.grid(axis="y", alpha=.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    saved += _save(fig, report_dir, "macro_f1", formats)

    # Desempenho por classe em todas as condições, sem esconder modelos.
    fig = _figure(figsize=(12, 4 * len(labels)))
    axes = fig.subplots(len(labels), 2, sharex=True, sharey=True, squeeze=False)
    for row_index, label_name in enumerate(labels):
        for column_index, metric_name in enumerate(("recall", "f1")):
            ax = axes[row_index][column_index]
            for series_index, (offset, (model_id, prompt_id)) in enumerate(zip(offsets, series)):
                color = colors[series_index % len(colors)]
                means, deviations = [], []
                for temperature in temperature_order:
                    values = [item[1]["per_class"][label_name][metric_name]
                              for item in grouped.get((model_id, temperature, prompt_id), [])]
                    means.append(statistics.mean(values) if values else np.nan)
                    deviations.append(statistics.pstdev(values) if len(values) > 1 else 0.0)
                ax.errorbar(x + offset, means, yerr=deviations, marker="o", capsize=3,
                            linewidth=1.5, label=series_label(model_id, prompt_id),
                            color=color)
            ax.set_title(f"{label_name} — {metric_name}")
            ax.set_ylim(0, 1)
            ax.grid(axis="y", alpha=.25)
            ax.set_xticks(x, [f"{temperature:g}" for temperature in temperature_order])
            if row_index == len(labels) - 1:
                ax.set_xlabel("Temperatura")
            if column_index == 0:
                ax.set_ylabel("Valor")
    axes[0][1].legend(fontsize=7, loc="lower left")
    fig.suptitle(f"Desempenho por classe — {stage}")
    fig.tight_layout()
    saved += _save(fig, report_dir, "per_class", formats)

    # Uma matriz por modelo e temperatura. Com repetições, exibe a média das
    # contagens por repetição (cada painel continua representando n notícias).
    predictions = (*labels, INVALID)
    short = _short(labels)
    for prompt_id in prompt_order:
        matrices = {}
        for model_id in model_order:
            for temperature in temperature_order:
                scores = [item[1] for item in grouped.get(
                    (model_id, temperature, prompt_id), [])]
                if not scores:
                    continue
                matrices[(model_id, temperature)] = np.mean([
                    [[score["confusion"][gold][pred] for pred in predictions] for gold in labels]
                    for score in scores
                ], axis=0)
        maximum = max((matrix.max() for matrix in matrices.values()), default=1)
        fig = _figure(figsize=(4 * len(temperature_order), 3.1 * len(model_order)))
        axes = fig.subplots(len(model_order), len(temperature_order), squeeze=False)
        image = None
        for row_index, model_id in enumerate(model_order):
            for column_index, temperature in enumerate(temperature_order):
                ax = axes[row_index][column_index]
                matrix = matrices.get((model_id, temperature))
                if matrix is None:
                    ax.axis("off")
                    continue
                image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=maximum)
                ax.set_xticks(range(len(predictions)), [*short, "Inv."])
                ax.set_yticks(range(len(labels)), short)
                ax.set_title(f"{model_label(model_id)} — T={temperature:g}", fontsize=9)
                if row_index == len(model_order) - 1:
                    ax.set_xlabel("Predição")
                if column_index == 0:
                    ax.set_ylabel("Rótulo do corpus")
                for i in range(len(labels)):
                    for j in range(len(predictions)):
                        value = matrix[i, j]
                        text_value = str(int(value)) if float(value).is_integer() else f"{value:.1f}"
                        color = "white" if value > maximum * .55 else "black"
                        ax.text(j, i, text_value, ha="center", va="center", color=color)
        fig.suptitle(f"Matrizes de confusão — {stage}, prompt {prompt_id}")
        fig.subplots_adjust(top=.91, right=.86, bottom=.08, hspace=.5, wspace=.35)
        if image is not None:
            color_axis = fig.add_axes([.9, .16, .018, .66])
            fig.colorbar(image, cax=color_axis, label="Média por repetição")
        stem = "confusion" if len(prompt_order) == 1 else f"confusion_prompt_{prompt_id}"
        saved += _save(fig, report_dir, stem, formats)

    stable_rows = [row for row in repeat_rows
                   if int(row["repetitions"]) > 1
                   and row["disagreement_fraction"] not in (None, "")]
    if stable_rows:
        fig = _figure(figsize=(9, 5))
        ax = fig.subplots()
        stable_temps = sorted({float(row["temperature"]) for row in stable_rows})
        stable_x = np.arange(len(stable_temps), dtype=float)
        width = .75 / max(len(series), 1)
        for index, (model_id, prompt_id) in enumerate(series):
            lookup = {(float(row["temperature"]), row["prompt"]):
                      float(row["disagreement_fraction"])
                      for row in stable_rows if row["model"] == model_id}
            values = [lookup.get((temperature, prompt_id), np.nan)
                      for temperature in stable_temps]
            ax.bar(stable_x - .375 + width / 2 + index * width,
                   values, width=width, label=series_label(model_id, prompt_id))
        ax.set_xticks(stable_x, [f"{temperature:g}" for temperature in stable_temps])
        ax.set_ylim(0, 1)
        ax.set_xlabel("Temperatura")
        ax.set_ylabel("Fração de notícias com desacordo")
        ax.set_title(f"Instabilidade entre repetições — {stage}")
        ax.grid(axis="y", alpha=.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        saved += _save(fig, report_dir, "stability", formats)

    stage_costs = [row for row in cost_rows if row.get("stage") == stage]
    if stage_costs:
        fig = _figure(figsize=(9, 5))
        ax = fig.subplots()
        cost_series = [(model_id, prompt_id) for model_id, prompt_id in series
                       if any(row["model"] == model_id and row["prompt_id"] == prompt_id
                              for row in stage_costs)]
        width = .75 / max(len(cost_series), 1)
        for index, (model_id, prompt_id) in enumerate(cost_series):
            lookup = {(float(row["temperature"]), row["prompt_id"]):
                      float(row["accounted_cost_usd"] or 0)
                      for row in stage_costs if row["model"] == model_id}
            values = [lookup.get((temperature, prompt_id), 0)
                      for temperature in temperature_order]
            ax.bar(x - .375 + width / 2 + index * width,
                   values, width=width, label=series_label(model_id, prompt_id))
        ax.set_xticks(x, [f"{temperature:g}" for temperature in temperature_order])
        ax.set_xlabel("Temperatura")
        ax.set_ylabel("Custo contabilizado (US$)")
        ax.set_title(f"Custo por modelo e temperatura — {stage}")
        ax.grid(axis="y", alpha=.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        saved += _save(fig, report_dir, "cost_by_condition", formats)
    return saved


def viability_figures(report_dir: Path, summary: list[dict],
                      formats: tuple[str, ...] = ("png",)) -> list[Path]:
    """Resultado das sondas básicas e custo por modelo; devolve os arquivos gravados."""
    import numpy as np

    _clear(report_dir, VIABILITY_FIGURES)
    if not summary:
        return []
    saved: list[Path] = []
    names = [model_label(row["model"]) for row in summary]
    y = np.arange(len(summary), dtype=float)

    fig = _figure(figsize=(9, 1.8 + .7 * len(summary)))
    ax = fig.subplots()
    left = np.zeros(len(summary))
    statuses = (("valid", "válida", "#2e7d32"), ("invalid", "inválida", "#f9a825"),
                ("model_mismatch", "modelo trocado", "#6a1b9a"),
                ("technical_errors", "erro técnico", "#c62828"))
    for field, label, color in statuses:
        values = np.array([float(row[field] or 0) for row in summary])
        ax.barh(y, values, left=left, color=color, label=label)
        left += values
    for index, row in enumerate(summary):
        mark = "aprovado" if str(row["base_approved"]) == "True" else "reprovado"
        ax.text(max(float(row["planned"] or 0), left[index]) + .1, index, mark, va="center")
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.set_xlim(0, max(float(r["planned"] or 0) for r in summary) + 2)
    ax.set_xlabel("Sondas")
    ax.set_title("Viabilidade — resultado das sondas por modelo")
    # Legenda abaixo do eixo, para não cobrir as marcações aprovado/reprovado.
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(.5, -.28), ncol=len(statuses),
              frameon=False)
    fig.tight_layout()
    saved += _save(fig, report_dir, "viability_status", formats)

    fig = _figure(figsize=(9, 5))
    ax = fig.subplots()
    width = .38
    x = np.arange(len(summary), dtype=float)
    ax.bar(x - width / 2, [float(r["provider_cost_usd"] or 0) for r in summary], width,
           label="informado pela DeepInfra")
    ax.bar(x + width / 2, [float(r["calculated_cost_usd"] or 0) for r in summary], width,
           label="estimativa local")
    ax.set_xticks(x, names)
    ax.set_ylabel("Custo (US$)")
    ax.set_title("Viabilidade — custo por modelo")
    ax.grid(axis="y", alpha=.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    saved += _save(fig, report_dir, "viability_cost", formats)
    return saved
