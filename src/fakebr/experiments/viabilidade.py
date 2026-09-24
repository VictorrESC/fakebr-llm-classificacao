"""Sondas técnicas da DeepInfra sem tocar na amostra do Fake.Br.

Cinco sondas básicas por modelo (três tamanhos de texto e três temperaturas) e,
opcionalmente, uma sonda de logprobs. Avalia formato, raciocínio exposto, uso e
custo; não mede acurácia.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from ..api import Executor, Outcome, Request, first_choice, request_key, request_payload, require_token
from ..cache import CacheSpec
from ..io import append_csv, canonical_hash, now_utc, read_csv, write_csv
from ..plots import output_formats, viability_figures
from ..prompts import INVALID
from ..project import Run
from .common import freeze_models, freeze_prompt_set, prompt_set, select_models

STAGES = ("run",)
FIRST_STAGES = ("run",)
REPORT_STAGES = ("run",)
DEFAULT_REPORT_STAGE = "run"
LEDGER_SOURCE = "viability"
RESULTS_FILE = "results.csv"

# Textos sintéticos neutros; não compõem a amostra nem as métricas de classificação.
CONTROLS = {
    "curto": (
        "A biblioteca municipal divulgou nesta terça-feira um novo horário de "
        "atendimento. A programação prevê abertura pela manhã e oficinas para "
        "moradores durante a semana."
    ),
    "medio": " ".join([
        "Em uma reunião pública, representantes de diferentes bairros conversaram "
        "sobre a agenda de eventos culturais do mês. Os participantes apresentaram "
        "propostas para ampliar o acesso às atividades e divulgar os horários."
    ] * 5),
    "longo": " ".join([
        "Uma associação comunitária publicou uma nota informativa sobre a preparação "
        "de uma feira de livros. O texto apresenta a programação, descreve os "
        "espaços disponíveis e convida a população a acompanhar novos comunicados."
    ] * 28),
}
BASE_PROBES = (("curto", 0.0), ("medio", 0.0), ("longo", 0.0), ("curto", 0.5), ("curto", 1.0))

RESULT_FIELDS = [
    "timestamp_utc", "probe", "control", "model_requested", "model_returned",
    "temperature", "request_sha256", "prompt_cache_key", "status", "prediction",
    "raw_response", "finish_reason", "reasoning_present", "logprobs_present",
    "logprobs_preview", "request_id", "attempts", "latency_seconds",
    "prompt_tokens", "cached_tokens", "completion_tokens", "provider_cost_usd",
    "calculated_cost_usd", "cost_source", "error_type", "http_status",
    "error", "sdk_version",
]
COST_FIELDS = [
    "timestamp_utc", "probe", "control", "model", "temperature",
    "request_sha256", "attempt", "status", "http_status", "prompt_tokens",
    "cached_tokens", "completion_tokens", "provider_cost_usd", "calculated_cost_usd",
    "cost_source", "request_id", "error_type",
]
SUMMARY_FIELDS = [
    "model", "planned", "completed", "valid", "invalid", "model_mismatch",
    "technical_errors", "reasoning_present", "usage_missing", "prompt_tokens",
    "completion_tokens", "cached_tokens", "provider_cost_usd",
    "calculated_cost_usd", "unknown_cost_attempts", "logprobs_supported",
    "logprobs_unsupported", "base_approved",
]
FINAL_STATUSES = {"valid", "invalid", "model_mismatch", "http_error",
                  "logprobs_supported", "logprobs_unsupported"}
CACHEABLE_STATUSES = frozenset({"valid", "invalid", "model_mismatch",
                                "logprobs_supported", "logprobs_unsupported"})
BLOCKING_STATUSES = {"model_mismatch", "http_error", "technical_error"}
CACHE = CacheSpec("viability", CACHEABLE_STATUSES, require_request_id=True)


def controls_fingerprint() -> dict:
    return {key: canonical_hash(value) for key, value in CONTROLS.items()}


def freeze(cfg: dict, root: Path) -> dict:
    experiment = cfg["experiment"]
    prompts = freeze_prompt_set(root, experiment["prompt_set"])
    if str(experiment["prompt"]) not in prompts["introductions"]:
        raise ValueError(f"experiment.prompt deve ser um de {list(prompts['introductions'])}")
    return {
        "runner": experiment["runner"], "description": experiment.get("description", ""),
        "prompt_set": prompts, "prompt": str(experiment["prompt"]),
        "models": freeze_models(root, experiment["models"]),
        "top_p": float(experiment["top_p"]), "max_tokens": int(experiment["max_tokens"]),
        "include_logprobs_probe": bool(experiment["include_logprobs_probe"]),
        "controls_sha256": controls_fingerprint(),
    }


def plans(experiment: dict, api: dict, models: list[dict] | None = None) -> list[dict]:
    if experiment["controls_sha256"] != controls_fingerprint():
        raise RuntimeError("Os textos de controle mudaram no código desde a criação deste run.")
    prompts, prompt_id = prompt_set(experiment), experiment["prompt"]
    probes = []
    for model in models or experiment["models"]:
        def payload(control: str, temperature: float) -> tuple[dict, str]:
            return request_payload(
                base_url=str(api["base_url"]), model=model, temperature=temperature,
                top_p=experiment["top_p"], max_tokens=experiment["max_tokens"],
                enable_prompt_cache_key=bool(api["enable_prompt_cache_key"]),
                prompt=prompts.render(prompt_id, CONTROLS[control]),
                shared_prefix=prompts.shared_prefix(prompt_id), repetition=1)

        for control, temperature in BASE_PROBES:
            body, key = payload(control, temperature)
            probes.append({"model": model, "control": control, "temperature": temperature,
                           "probe": "base", "body": body, "key": key})
        if experiment["include_logprobs_probe"]:
            body, _ = payload("curto", 0.0)
            body = {**body, "logprobs": True, "top_logprobs": 20}
            probes.append({"model": model, "control": "curto", "temperature": 0.,
                           "probe": "logprobs", "body": body,
                           "key": request_key(str(api["base_url"]), body, 1)})
    return probes


def preview_logprobs(choice) -> tuple[bool, str]:
    logprobs = getattr(choice, "logprobs", None)
    tokens = getattr(logprobs, "content", None) if logprobs else None
    if not tokens:
        return False, ""
    top = getattr(tokens[0], "top_logprobs", None) or []
    sample = [{"token": str(item.token), "logprob": item.logprob} for item in top[:20]]
    return True, json.dumps(sample, ensure_ascii=False)


def summarize(out: Path, planned: list[dict]) -> list[dict]:
    responses = {r["request_sha256"]: r for r in read_csv(out / RESULTS_FILE)}
    ledger = read_csv(out / "costs.csv")
    rows = []
    for model_id in dict.fromkeys(p["model"]["id"] for p in planned):
        expected = [p for p in planned if p["model"]["id"] == model_id]
        base = [responses[p["key"]] for p in expected
                if p["probe"] == "base" and p["key"] in responses]
        extra = [responses[p["key"]] for p in expected
                 if p["probe"] == "logprobs" and p["key"] in responses]
        charges = [r for r in ledger if r["model"] == model_id]

        def total(field: str) -> int:
            return sum(int(r.get(field) or 0) for r in charges)

        rows.append({
            "model": model_id, "planned": len(expected),
            "completed": len(base), "valid": sum(r["status"] == "valid" for r in base),
            "invalid": sum(r["status"] == "invalid" for r in base),
            "model_mismatch": sum(r["status"] == "model_mismatch" for r in base),
            "technical_errors": sum(r["status"] not in {"valid", "invalid", "model_mismatch"}
                                    for r in base),
            "reasoning_present": sum(r.get("reasoning_present") == "True" for r in base),
            "usage_missing": sum(not r.get("cost_source") or
                                 r["cost_source"] == "unknown" for r in base),
            "prompt_tokens": total("prompt_tokens"),
            "completion_tokens": total("completion_tokens"),
            "cached_tokens": total("cached_tokens"),
            "provider_cost_usd": round(sum(float(r.get("provider_cost_usd") or 0)
                                           for r in charges), 10),
            "calculated_cost_usd": round(sum(float(r.get("calculated_cost_usd") or 0)
                                             for r in charges), 10),
            "unknown_cost_attempts": sum(r.get("cost_source") == "unknown" for r in charges),
            "logprobs_supported": sum(r["status"] == "logprobs_supported" for r in extra),
            "logprobs_unsupported": sum(r["status"] == "logprobs_unsupported" for r in extra),
            "base_approved": (len(base) == len(BASE_PROBES) and all(
                r["status"] == "valid" and r.get("finish_reason") == "stop"
                and r.get("reasoning_present") == "False"
                and r.get("cost_source") != "unknown" for r in base)),
        })
    write_csv(out / "summary.csv", rows, SUMMARY_FIELDS)
    return rows


def run(run: Run, stage: str) -> None:
    import openai

    out = run.out
    planned = plans(run.experiment, run.api, select_models(run.experiment["models"],
                                                          run.select.get("models")))
    central = run.ledger(LEDGER_SOURCE)
    central.add_many(read_csv(out / "costs.csv"))
    cache = run.cache(CACHE)
    existing_rows = read_csv(out / RESULTS_FILE)
    for existing_row in existing_rows:
        cache.put(existing_row)
    finished = {r["request_sha256"] for r in existing_rows if r["status"] in FINAL_STATUSES}
    reused = 0
    for item in planned:
        if item["key"] in finished:
            continue
        cached = cache.get(item["key"])
        if cached is None:
            continue
        if cached.get("model_requested") != item["model"]["id"]:
            raise RuntimeError(f"Cache incompatível para {item['key']}")
        copied = {field: cached.get(field, "") for field in RESULT_FIELDS}
        copied.update({"timestamp_utc": now_utc(), "attempts": 0,
                       "latency_seconds": 0, "provider_cost_usd": 0,
                       "calculated_cost_usd": 0,
                       "cost_source": ("unknown" if cached.get("cost_source") == "unknown"
                                       else "reused")})
        append_csv(out / RESULTS_FILE, copied, RESULT_FIELDS)
        finished.add(item["key"])
        reused += 1
    prior = {r["request_sha256"]: r for r in read_csv(out / RESULTS_FILE)}
    # O primeiro resultado de um modelo com ID trocado ou HTTP permanente bloqueia o resto.
    blocked_prior = {p["model"]["id"] for p in planned
                     if p["probe"] == "base" and p["control"] == "curto"
                     and p["temperature"] == 0.0 and p["key"] in prior
                     and prior[p["key"]]["status"] in {"model_mismatch", "http_error"}}
    pending = [p for p in planned if p["key"] not in finished
               and p["model"]["id"] not in blocked_prior]
    print(f"Viabilidade: {len(pending)} chamadas novas; {reused} reutilizadas; "
          f"{len(planned)} planejadas.", flush=True)
    if blocked_prior:
        print(f"Modelos bloqueados por primeiro resultado anterior: {sorted(blocked_prior)}",
              flush=True)
    if not pending:
        summarize(out, planned)
        return

    token = require_token(run.api)
    prompts = prompt_set(run.experiment)

    def record_cost(ledger: dict) -> None:
        append_csv(out / "costs.csv", ledger, COST_FIELDS)
        central.add(ledger)

    def finish(outcome: Outcome) -> dict:
        p = outcome.request.data
        model_id = p["model"]["id"]
        if outcome.ok:
            response = outcome.response
            message, raw, finish_reason = first_choice(response)
            choices = getattr(response, "choices", None) or []
            reasoning = getattr(message, "reasoning_content", None) if message else None
            leaked = bool(reasoning) or "<think" in raw.lower() or "</think" in raw.lower()
            has_logs, logs = preview_logprobs(choices[0]) if choices else (False, "")
            returned = getattr(response, "model", "") or ""
            prediction = prompts.parse(raw)
            if returned != model_id:
                status = "model_mismatch"
            elif p["probe"] == "logprobs":
                status = "logprobs_supported" if has_logs else "logprobs_unsupported"
            else:
                status = ("valid" if prediction != INVALID and not leaked
                          and finish_reason == "stop" else "invalid")
            result = {"status": status, "model_returned": returned,
                      "prediction": prediction, "raw_response": raw,
                      "reasoning_present": leaked, "finish_reason": finish_reason,
                      "logprobs_present": has_logs, "logprobs_preview": logs,
                      "request_id": getattr(response, "id", "") or "", **outcome.cost}
        else:
            http = outcome.http_status
            status = ("logprobs_unsupported" if p["probe"] == "logprobs" and http in (400, 422)
                      else "http_error" if http is not None and not outcome.transient
                      else "technical_error")
            result = {"status": status, "http_status": http or "",
                      "error_type": outcome.error_type, "error": outcome.error_text[:350]}
        saved = {"timestamp_utc": now_utc(), "probe": p["probe"], "control": p["control"],
                 "model_requested": model_id, "temperature": p["temperature"],
                 "request_sha256": p["key"],
                 "prompt_cache_key": p["body"].get("extra_body", {}).get("prompt_cache_key", ""),
                 "sdk_version": openai.__version__, "attempts": outcome.attempts,
                 "latency_seconds": outcome.latency_seconds, **result}
        append_csv(out / RESULTS_FILE, saved, RESULT_FIELDS)
        cache.put(saved)
        print(f"{model_id} {p['probe']} {p['control']} T={p['temperature']}: {status}", flush=True)
        return saved

    requests = [Request(key=p["key"], body=p["body"], pricing=p["model"]["pricing"],
                        group=(p["model"]["id"],),
                        ledger={"probe": p["probe"], "control": p["control"],
                                "model": p["model"]["id"], "temperature": p["temperature"]},
                        data=p) for p in pending]
    executor = Executor(run.api, token, record_cost, label="viabilidade", log=lambda _: None)
    try:
        result = asyncio.run(executor.run(
            requests, finish, blocks_group=lambda row, _: row["status"] in BLOCKING_STATUSES))
    finally:
        summarize(out, planned)
    if result.stopped:
        raise RuntimeError(f"Execução interrompida: {result.stopped}")
    print(f"Arquivos: {out / RESULTS_FILE}, {out / 'costs.csv'}, {out / 'summary.csv'}")


def report(run: Run, stage: str) -> dict:
    """Resumo por modelo e figuras em runs/<run>/report/."""
    formats = output_formats(run.report)
    summary = summarize(run.out, plans(run.experiment, run.api))
    report_dir = run.out / "report"
    report_dir.mkdir(exist_ok=True)
    saved = viability_figures(report_dir, summary, formats)
    print(f"Resumo: {run.out / 'summary.csv'}; figuras: {report_dir}")
    return {"report_dir": report_dir,
            "tables": [run.out / "summary.csv", run.out / RESULTS_FILE],
            "figures": saved}
