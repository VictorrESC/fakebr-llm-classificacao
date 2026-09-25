"""Classificação de corpus pareado por vários modelos, temperaturas, prompts e repetições.

Etapas: ``prepare`` sorteia o manifesto sem chamar a API; ``dev`` e ``eval``
executam as amostras correspondentes e podem ser retomadas.
"""

from __future__ import annotations

import asyncio
import random
import statistics
import time
from collections import defaultdict
from pathlib import Path

from .. import data
from ..api import Executor, Outcome, Request, first_choice, request_payload, require_token
from ..cache import CacheSpec
from ..io import append_csv, atomic_json, now_utc, read_csv, sha256, write_csv
from ..metrics import apply_length_rule, fit_length_rule, pair_bootstrap, summarize
from ..plots import condition_name, condition_parts, figures, output_formats
from ..prompts import INVALID
from ..project import Run
from .common import freeze_models, freeze_prompt_set, prompt_set, select_models, selected

STAGES = ("prepare", "dev", "eval")
FIRST_STAGES = ("prepare",)
REPORT_STAGES = ("dev", "eval")
DEFAULT_REPORT_STAGE = "dev"
LEDGER_SOURCE = "pilot"
RESULTS_FILE = "responses.csv"
CACHE = CacheSpec("responses")
LOADERS = ("fakebr_pairs",)

RESPONSE_FIELDS = [
    "timestamp_utc", "stage", "model_requested", "temperature", "prompt_id",
    "repetition", "document_id", "pair_id",
    "gold_label", "text_sha256", "characters", "request_sha256",
    "prompt_cache_key", "model_returned", "status",
    "prediction", "raw_response", "finish_reason", "request_id", "attempts",
    "latency_seconds", "prompt_tokens", "cached_tokens", "completion_tokens",
    "provider_cost_usd", "calculated_cost_usd", "cost_source", "error_type",
    "error", "sdk_version",
]
LEDGER_FIELDS = [
    "timestamp_utc", "stage", "model", "temperature", "prompt_id",
    "repetition", "document_id",
    "request_sha256", "attempt", "status", "http_status",
    "prompt_tokens", "cached_tokens", "completion_tokens", "provider_cost_usd",
    "calculated_cost_usd", "cost_source", "error_type", "request_id",
]
COST_SUMMARY_FIELDS = [
    "stage", "prompt_id", "model", "temperature", "attempts", "successful",
    "error_attempts", "unknown_cost_attempts", "prompt_tokens",
    "cached_tokens", "completion_tokens", "provider_cost_usd",
    "calculated_cost_usd", "accounted_cost_usd",
]


# --- Configuração -------------------------------------------------------------

def freeze(cfg: dict, root: Path) -> dict:
    experiment = cfg["experiment"]
    dataset_name = experiment["dataset"]["name"]
    if dataset_name not in cfg.get("datasets", {}):
        raise ValueError(f"Corpus desconhecido: {dataset_name} (veja datasets em conf/config.yaml)")
    source = cfg["datasets"][dataset_name]
    texts = str(experiment["dataset"].get("texts", "full_texts"))
    if source.get("loader", "fakebr_pairs") not in LOADERS:
        raise ValueError(f"Leitor de corpus não suportado: {source.get('loader')}")
    prompts = freeze_prompt_set(root, experiment["prompt_set"])
    planned_prompts = [str(p) for p in experiment["prompts"]]
    if (not planned_prompts or len(set(planned_prompts)) != len(planned_prompts)
            or any(p not in prompts["introductions"] for p in planned_prompts)):
        raise ValueError(f"experiment.prompts deve usar, sem repetição, {list(prompts['introductions'])}")
    conditions = [{"temperature": float(c["temperature"]), "repetitions": int(c["repetitions"])}
                  for c in experiment["conditions"]]
    if not conditions or any(c["repetitions"] < 1 or not 0 <= c["temperature"] <= 2
                             for c in conditions):
        raise ValueError("Condições inválidas: temperatura entre 0 e 2, repetições positivas")
    if len({c["temperature"] for c in conditions}) != len(conditions):
        raise ValueError("Há temperaturas repetidas em experiment.conditions")
    if int(experiment["dev_pairs"]) < 0 or int(experiment["eval_pairs"]) < 0:
        raise ValueError("dev_pairs e eval_pairs não podem ser negativos")
    pairs_from = experiment.get("pairs_from")
    if pairs_from is not None and (not isinstance(pairs_from, str) or not pairs_from):
        raise ValueError("experiment.pairs_from deve ser o nome de um run em runs/ ou null")
    return {
        "runner": experiment["runner"], "description": experiment.get("description", ""),
        "dataset": {"name": dataset_name, "loader": source.get("loader", "fakebr_pairs"),
                    "dir": source["dir"], "texts": texts, "revision": source.get("revision"),
                    "fingerprint": (source.get("fingerprint") or {}).get(texts)},
        "seed": int(experiment["seed"]),
        "dev_pairs": int(experiment["dev_pairs"]), "eval_pairs": int(experiment["eval_pairs"]),
        "pairs_from": pairs_from,
        "prompt_set": prompts, "prompts": planned_prompts,
        "models": freeze_models(root, experiment["models"]),
        "conditions": conditions,
        "top_p": float(experiment["top_p"]), "max_tokens": int(experiment["max_tokens"]),
        "bootstrap_repeats": int(experiment.get("bootstrap_repeats", 500)),
    }


def _plan_axes(run: Run) -> tuple[list[dict], list[dict], list[str]]:
    experiment = run.experiment
    models = select_models(experiment["models"], run.select.get("models"))
    temperatures = run.select.get("temperatures")
    conditions = selected(experiment["conditions"],
                          None if temperatures is None else [float(t) for t in temperatures],
                          "temperatures", key=lambda c: float(c["temperature"]))
    prompts = selected(experiment["prompts"], run.select.get("prompts"), "prompts")
    return models, conditions, prompts


def _dataset_dir(run: Run) -> Path:
    return run.paths.data / run.experiment["dataset"]["dir"]


def _stage_rows(run: Run, stage: str) -> tuple[list[dict], dict[str, str]]:
    """Linhas do manifesto da etapa e textos, conferindo o hash de cada um."""
    manifest = read_csv(run.out / "manifest.csv")
    if not manifest:
        raise FileNotFoundError(f"{run.out / 'manifest.csv'} ausente: execute --stage prepare.")
    rows = [row for row in manifest if row["stage"] == stage]
    if not rows:
        raise ValueError(f"Etapa vazia ou desconhecida: {stage}")
    dataset = _dataset_dir(run)
    if not dataset.is_dir():
        raise FileNotFoundError(f"Dataset não encontrado: {dataset}. Rode 'pixi run fetch-data'.")
    texts = {}
    for row in rows:
        content = data.read_text(dataset / row["relative_path"])
        if sha256(content) != row["text_sha256"]:
            raise RuntimeError(f"Texto alterado: {row['relative_path']}")
        texts[row["relative_path"]] = content
    return rows, texts


def _plan(run: Run, rows: list[dict], texts: dict[str, str]) -> list[dict]:
    experiment, prompts_def = run.experiment, prompt_set(run.experiment)
    models, conditions, prompts = _plan_axes(run)
    items = []
    for model in models:
        for condition in conditions:
            temperature = float(condition["temperature"])
            for prompt_id in prompts:
                prefix = prompts_def.shared_prefix(prompt_id)
                for repetition in range(1, int(condition["repetitions"]) + 1):
                    for row in rows:
                        body, key = request_payload(
                            base_url=str(run.api["base_url"]), model=model,
                            temperature=temperature, top_p=experiment["top_p"],
                            max_tokens=experiment["max_tokens"],
                            enable_prompt_cache_key=bool(run.api["enable_prompt_cache_key"]),
                            prompt=prompts_def.render(prompt_id, texts[row["relative_path"]]),
                            shared_prefix=prefix, repetition=repetition)
                        items.append({"model": model, "temperature": temperature,
                                      "prompt_id": prompt_id, "repetition": repetition,
                                      "row": row, "body": body, "key": key})
    return items


# --- Execução -----------------------------------------------------------------

def run(run: Run, stage: str) -> None:
    if stage == "prepare":
        prepare(run)
        return
    rows, texts = _stage_rows(run, stage)
    plan = _plan(run, rows, texts)
    out = run.out
    central = run.ledger(LEDGER_SOURCE)
    central.add_many(read_csv(out / "costs.csv"))
    cache = run.cache(CACHE)
    existing_rows = read_csv(out / RESULTS_FILE)
    for existing_row in existing_rows:
        cache.put(existing_row)
    already = {r["request_sha256"] for r in existing_rows if r.get("status") == "ok"}

    pending, reused = [], 0
    for item in plan:
        if item["key"] in already:
            continue
        cached = cache.get(item["key"])
        if cached is None:
            pending.append(item)
            continue
        row = item["row"]
        if (cached.get("text_sha256") != row["text_sha256"]
                or cached.get("model_requested") != item["model"]["id"]):
            raise RuntimeError(f"Cache incompatível para {item['key']}")
        copied = {field: cached.get(field, "") for field in RESPONSE_FIELDS}
        copied.update({"timestamp_utc": now_utc(), "stage": stage,
                       "prompt_id": item["prompt_id"], "repetition": item["repetition"],
                       "document_id": f"{row['gold_label']}:{row['pair_id']}",
                       "pair_id": row["pair_id"], "gold_label": row["gold_label"],
                       "characters": row["characters"], "request_sha256": item["key"],
                       "attempts": 0, "latency_seconds": 0,
                       "cost_source": ("unknown" if cached.get("cost_source") == "unknown"
                                       else "reused"),
                       "provider_cost_usd": 0, "calculated_cost_usd": 0})
        append_csv(out / RESULTS_FILE, copied, RESPONSE_FIELDS)
        already.add(item["key"])
        reused += 1
    random.Random(int(run.experiment["seed"]) + (stage == "eval")).shuffle(pending)
    print(f"{stage}: {len(pending)} requisições novas; {reused} reutilizadas; "
          f"{len(plan)} planejadas.", flush=True)
    if not pending:
        cost_summary(out)
        return

    import openai

    token = require_token(run.api)
    labels = prompt_set(run.experiment)

    def record_cost(ledger: dict) -> None:
        append_csv(out / "costs.csv", ledger, LEDGER_FIELDS)
        central.add(ledger)

    def finish(outcome: Outcome) -> dict:
        item = outcome.request.data
        row = item["row"]
        if outcome.ok:
            _, raw, finish_reason = first_choice(outcome.response)
            has_choice = bool(getattr(outcome.response, "choices", None))
            result = {"status": "ok" if has_choice else "technical_error",
                      "prediction": labels.parse(raw) if has_choice else "",
                      "raw_response": raw, "finish_reason": finish_reason or "",
                      "model_returned": getattr(outcome.response, "model", "") or "",
                      "request_id": getattr(outcome.response, "id", "") or "",
                      "error_type": "" if has_choice else "EmptyChoices",
                      **outcome.cost}
        else:
            result = {"status": "technical_error", "error_type": outcome.error_type,
                      "error": outcome.error_text[:300]}
        saved = {
            "timestamp_utc": now_utc(), **outcome.request.ledger,
            "request_sha256": item["key"], "pair_id": row["pair_id"],
            "gold_label": row["gold_label"], "text_sha256": row["text_sha256"],
            "characters": row["characters"],
            "prompt_cache_key": item["body"].get("extra_body", {}).get("prompt_cache_key", ""),
            "model_requested": item["model"]["id"], "sdk_version": openai.__version__,
            "attempts": outcome.attempts, "latency_seconds": outcome.latency_seconds, **result,
        }
        append_csv(out / RESULTS_FILE, saved, RESPONSE_FIELDS)
        cache.put(saved)
        return saved

    requests = [Request(
        key=item["key"], body=item["body"], pricing=item["model"]["pricing"],
        group=(item["model"]["id"], item["temperature"]),
        ledger={"stage": stage, "model": item["model"]["id"], "temperature": item["temperature"],
                "prompt_id": item["prompt_id"], "repetition": item["repetition"],
                "document_id": f"{item['row']['gold_label']}:{item['row']['pair_id']}"},
        data=item) for item in pending]
    executor = Executor(run.api, token, record_cost, label=stage)
    started = time.monotonic()
    try:
        result = asyncio.run(executor.run(requests, finish))
    finally:
        cost_summary(out)
    elapsed = max(0.001, time.monotonic() - started)
    print(f"{result.completed} respostas em {elapsed:.1f}s "
          f"({result.completed / elapsed:.1f} respostas/s, média). "
          f"Respostas: {out / RESULTS_FILE}; custos: {out / 'costs.csv'}")
    _raise_if_interrupted(result)


def _raise_if_interrupted(result) -> None:
    problems = []
    if result.stopped:
        problems.append(f"execução interrompida ({result.stopped})")
    problems += [f"{model} T={temperature:g} bloqueado ({reason})"
                 for (model, temperature), reason in result.blocked.items()]
    if problems:
        raise RuntimeError("Requisições não concluídas: " + "; ".join(problems)
                           + ". Confira responses.csv e costs.csv antes de repetir.")


def prepare(run: Run) -> None:
    if (run.out / "manifest.csv").exists():
        raise FileExistsError(f"{run.name} já foi preparado; use outro --run para um novo sorteio.")
    experiment = run.experiment
    dataset = _dataset_dir(run)
    if not dataset.is_dir():
        raise FileNotFoundError(f"Dataset não encontrado: {dataset}. Rode 'pixi run fetch-data'.")
    source = experiment.get("pairs_from")
    pairs, source_info = _source_pairs(run, source) if source else (None, None)
    rows, info = data.build_manifest(
        dataset, experiment["dataset"]["texts"], prompt_set(experiment).labels,
        experiment["seed"], experiment["dev_pairs"], experiment["eval_pairs"],
        experiment["dataset"].get("fingerprint"), pairs=pairs)
    if source_info:
        info["pairs_from"] = source_info
    write_csv(run.out / "manifest.csv", rows, list(rows[0]))
    atomic_json(run.out / "manifest.json", info)
    origin = f" Pares de {source}." if source else ""
    print(f"Manifesto pronto: dev={experiment['dev_pairs'] * 2}, "
          f"eval={experiment['eval_pairs'] * 2} notícias.{origin} "
          f"Pares excluídos={len(info['excluded_pairs'])}. Pasta: {run.out}")


def _source_pairs(run: Run, source: str) -> tuple[dict[str, list[str]], dict]:
    """Pares dev/eval do manifesto de outro run, na ordem em que aparecem."""
    path = run.paths.run_dir(source) / "manifest.csv"
    if not path.is_file():
        raise FileNotFoundError(f"experiment.pairs_from: {path} não existe.")
    pairs: dict[str, list[str]] = {"dev": [], "eval": []}
    for row in read_csv(path):
        stage_pairs = pairs.setdefault(row["stage"], [])
        if row["pair_id"] not in stage_pairs:
            stage_pairs.append(row["pair_id"])
    return pairs, {"run": source,
                   "manifest_sha256": sha256(path.read_text(encoding="utf-8"))}


def cost_summary(out: Path) -> None:
    groups = defaultdict(lambda: {"attempts": 0, "successful": 0, "error_attempts": 0,
                                  "unknown_cost_attempts": 0, "prompt_tokens": 0,
                                  "cached_tokens": 0, "completion_tokens": 0,
                                  "provider_cost_usd": 0., "calculated_cost_usd": 0.,
                                  "accounted_cost_usd": 0.})
    for row in read_csv(out / "costs.csv"):
        group = groups[(row["stage"], row["prompt_id"], row["model"], row["temperature"])]
        group["attempts"] += 1
        group["successful"] += row["status"] == "ok"
        group["error_attempts"] += row["status"] != "ok"
        group["unknown_cost_attempts"] += row.get("cost_source") == "unknown"
        for field in ("prompt_tokens", "cached_tokens", "completion_tokens"):
            group[field] += int(row.get(field) or 0)
        for field in ("provider_cost_usd", "calculated_cost_usd"):
            group[field] += float(row.get(field) or 0)
        provider = row.get("provider_cost_usd", "")
        group["accounted_cost_usd"] += float(provider if provider != ""
                                             else row.get("calculated_cost_usd") or 0)
    summary = [{"stage": stage, "prompt_id": prompt_id, "model": model,
                "temperature": temperature,
                **{k: round(v, 9) if "cost_usd" in k else v for k, v in values.items()}}
               for (stage, prompt_id, model, temperature), values in sorted(groups.items())]
    write_csv(out / "cost_summary.csv", summary, COST_SUMMARY_FIELDS)


# --- Relatório ----------------------------------------------------------------

def report(run: Run, stage: str) -> dict:
    """Métricas, tabelas e figuras da etapa; devolve a pasta e os arquivos gerados."""
    experiment, out = run.experiment, run.out
    formats = output_formats(run.report)
    labels = prompt_set(experiment).answers
    rows_for_stage, texts = _stage_rows(run, stage)
    if len(rows_for_stage) != experiment[f"{stage}_pairs"] * 2:
        raise ValueError(f"Manifesto {stage} incompleto")
    latest = {r["request_sha256"]: r for r in read_csv(out / RESULTS_FILE)}
    all_rows, grouped = {}, defaultdict(list)
    missing_total = []
    for item in _plan(run, rows_for_stage, texts):
        name = condition_name(item["model"]["id"], item["temperature"],
                              item["repetition"], item["prompt_id"])
        found = latest.get(item["key"])
        if not found or found["status"] != "ok":
            missing_total.append(name)
            continue
        if name not in all_rows:
            all_rows[name] = []
            grouped[(item["model"]["id"], item["temperature"], item["prompt_id"])].append(name)
        all_rows[name].append(found)
    if missing_total:
        first = missing_total[0]
        raise RuntimeError(f"{first}: faltam {missing_total.count(first)} respostas "
                           f"({len(missing_total)} no total). "
                           f"Repita --stage {stage} antes do relatório.")

    report_dir = out / f"report_{stage}"
    report_dir.mkdir(exist_ok=True)
    metrics, metric_rows, class_rows, confusion_rows = {}, [], [], []
    for name, rows in all_rows.items():
        score = summarize(rows, labels)
        metrics[name] = score
        model_id, temperature, repetition, prompt_id = condition_parts(name)
        ci = (pair_bootstrap(rows, labels, experiment["seed"] + 2000,
                             repeats=experiment["bootstrap_repeats"])
              if stage == "eval" and temperature == 0 else ["", ""])
        metric_rows.append({"model": model_id, "temperature": temperature,
                            "repetition": repetition, "prompt": prompt_id,
                            "n": score["n"], "valid_rate": score["valid_rate"],
                            "accuracy": score["accuracy"],
                            "balanced_accuracy": score["balanced_accuracy"],
                            "macro_f1": score["macro_f1"],
                            "mcc_valid_only": score["mcc_valid_only"],
                            "macro_f1_ci95_low": ci[0], "macro_f1_ci95_high": ci[1]})
        for label in labels:
            class_rows.append({"condition": name, "class": label, **score["per_class"][label]})
            confusion_rows.append({"condition": name, "gold_label": label,
                                   **score["confusion"][label]})
    atomic_json(report_dir / "metrics.json", metrics)
    write_csv(report_dir / "metrics.csv", metric_rows,
              ["model", "temperature", "repetition", "prompt", "n", "valid_rate",
               "accuracy", "balanced_accuracy", "macro_f1", "mcc_valid_only",
               "macro_f1_ci95_low", "macro_f1_ci95_high"])
    write_csv(report_dir / "per_class.csv", class_rows,
              ["condition", "class", "support", "precision", "recall", "f1"])
    write_csv(report_dir / "confusion.csv", confusion_rows,
              ["condition", "gold_label", *labels, INVALID])
    summary = []
    for (model_id, temperature, prompt_id), names in grouped.items():
        f1s = [metrics[name]["macro_f1"] for name in names]
        by_document = defaultdict(set)
        for name in names:
            for row in all_rows[name]:
                by_document[row["document_id"]].add(row["prediction"])
        summary.append({"model": model_id, "temperature": temperature, "prompt": prompt_id,
                        "repetitions": len(names), "mean_macro_f1": statistics.mean(f1s),
                        "std_macro_f1": statistics.pstdev(f1s),
                        "min_macro_f1": min(f1s), "max_macro_f1": max(f1s),
                        "disagreement_fraction": (sum(len(x) > 1 for x in by_document.values())
                                                  / len(by_document) if len(names) > 1 else "")})
    write_csv(report_dir / "repeat_summary.csv", summary,
              ["model", "temperature", "prompt", "repetitions", "mean_macro_f1",
               "std_macro_f1", "min_macro_f1", "max_macro_f1", "disagreement_fraction"])
    tables = ["metrics.csv", "per_class.csv", "confusion.csv", "repeat_summary.csv"]
    baseline = _length_baseline(run, stage, rows_for_stage, labels)
    if baseline:
        write_csv(report_dir / "length_baseline.csv", [baseline], list(baseline))
        tables.append("length_baseline.csv")
        print(f"Baseline de comprimento (palavras > {baseline['threshold_words']} => "
              f"{baseline['above_label']}): acurácia {baseline['accuracy']:.3f}, "
              f"Macro-F1 {baseline['macro_f1']:.3f}")
    else:
        (report_dir / "length_baseline.csv").unlink(missing_ok=True)
    cost_summary(out)
    saved = (figures(report_dir, metrics, summary, read_csv(out / "cost_summary.csv"), stage,
                     labels, formats, baseline_macro_f1=baseline and baseline["macro_f1"])
             if all_rows else [])
    print(f"Relatório {stage}: {len(metric_rows)} condições. Métricas: {report_dir / 'metrics.csv'}")
    return {"report_dir": report_dir,
            "tables": [report_dir / name for name in tables] + [out / "cost_summary.csv"],
            "figures": saved}


def _length_baseline(run: Run, stage: str, rows: list[dict],
                     labels: tuple[str, ...]) -> dict | None:
    """Regra "palavras > limiar" ajustada nos pares do corpus fora da amostra do run.

    Nenhum par de dev/eval entra no ajuste, então o limiar não vê a amostra avaliada.
    Devolve ``None`` com mais de dois rótulos ou sem pares fora da amostra.
    """
    if len(labels) != 2:
        return None
    experiment = run.experiment
    sampled = {row["pair_id"] for row in read_csv(run.out / "manifest.csv")}
    counts = data.word_counts(_dataset_dir(run), experiment["dataset"]["texts"],
                              prompt_set(experiment).labels)
    train = [(words, label) for pair_id, by_label in counts.items() if pair_id not in sampled
             for label, words in by_label.items()]
    if not train:
        return None
    rule = fit_length_rule(train, labels)
    predicted = [{"pair_id": row["pair_id"], "gold_label": row["gold_label"],
                  "prediction": apply_length_rule(rule, int(row["words"]))} for row in rows]
    score = summarize(predicted, labels)
    ci = (pair_bootstrap(predicted, labels, experiment["seed"] + 2000,
                         repeats=experiment["bootstrap_repeats"])
          if stage == "eval" else ["", ""])
    return {**rule, "fit_pairs": len(train) // 2, "n": score["n"],
            "accuracy": score["accuracy"], "balanced_accuracy": score["balanced_accuracy"],
            "macro_f1": score["macro_f1"], "mcc_valid_only": score["mcc_valid_only"],
            "macro_f1_ci95_low": ci[0], "macro_f1_ci95_high": ci[1]}
