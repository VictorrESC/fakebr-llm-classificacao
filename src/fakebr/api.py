"""Montagem de requisições, custos e execução concorrente na API DeepInfra.

Todos os experimentos usam o mesmo executor: limite de taxa adaptativo,
novas tentativas para erros transitórios, uma linha no livro-caixa por
tentativa (antes de interpretar a resposta) e bloqueio por grupo quando a
configuração de um modelo se mostra inválida.
"""

from __future__ import annotations

import asyncio
import os
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .io import canonical_hash, now_utc


TRANSIENT_STATUS = frozenset({429, 500, 502, 503, 504})
# Autenticação inválida interrompe tudo; modelo inexistente/proibido, só o grupo.
STOP_ALL_STATUS = frozenset({401})
STOP_GROUP_STATUS = frozenset({403, 404})
RUNTIME_KEYS = ("token_env", "start_rps", "target_rps", "ramp_step_rps",
                "ramp_every_successes", "concurrency", "timeout_seconds", "max_attempts")
REQUEST_KEYS = ("base_url", "enable_prompt_cache_key")


def request_payload(*, base_url: str, model: dict, temperature: float, top_p: float,
                    max_tokens: int, enable_prompt_cache_key: bool, prompt: str,
                    shared_prefix: str, repetition: int) -> tuple[dict, str]:
    """Corpo da requisição e sua chave SHA-256 (cache e identificação de respostas).

    Qualquer mudança aqui altera as chaves e invalida o cache e os runs antigos;
    tests/test_compat.py protege os valores.
    """
    extra_body = model.get("extra_body") or {}
    body = {
        "model": model["id"],
        "messages": [{"role": "user", "content": prompt}],
        "temperature": float(temperature), "top_p": float(top_p),
        "max_tokens": int(max_tokens), "n": 1, "stream": False,
    }
    extra = dict(extra_body)
    if enable_prompt_cache_key:
        # Prefixo compartilhado: jamais inserir o ID da notícia aqui.
        extra["prompt_cache_key"] = canonical_hash({
            "model": model["id"], "prefix": shared_prefix,
            "temperature": float(temperature), "top_p": float(top_p),
            "max_tokens": int(max_tokens), "model_extra_body": extra_body,
            "repetition": repetition,
        })
    if extra:
        body["extra_body"] = extra
    # A repetição identifica a medição: inclui-a na chave mesmo com payload igual.
    return body, request_key(base_url, body, repetition)


def request_key(base_url: str, body: dict, repetition: int) -> str:
    return canonical_hash({"base_url": base_url, "body": body, "repetition": repetition})


def cost_fields(usage: dict, pricing: dict) -> dict:
    prompt = int(usage.get("prompt_tokens") or 0)
    completion = int(usage.get("completion_tokens") or 0)
    details = usage.get("prompt_tokens_details") or {}
    cached = min(int(details.get("cached_tokens") or 0), prompt)
    # Tarifa de cache ausente: usar tarifa normal, evitando desconto presumido.
    cached_rate = (float(pricing["cached_usd_per_million"])
                   if pricing.get("cached_usd_per_million") is not None else
                   float(pricing["input_usd_per_million"]))
    calculated = ((prompt - cached) * float(pricing["input_usd_per_million"])
                  + cached * cached_rate
                  + completion * float(pricing["output_usd_per_million"])) / 1_000_000
    provider = usage.get("estimated_cost")
    return {
        "prompt_tokens": prompt, "cached_tokens": cached,
        "completion_tokens": completion,
        "provider_cost_usd": provider if provider is not None else "",
        "calculated_cost_usd": round(calculated, 10),
        "cost_source": "provider" if provider is not None else "local_estimate",
    }


UNKNOWN_COST = {"prompt_tokens": "", "cached_tokens": "", "completion_tokens": "",
                "provider_cost_usd": "", "calculated_cost_usd": "", "cost_source": "unknown"}


def first_choice(response: Any) -> tuple[Any, str, str | None]:
    """(mensagem, conteúdo, finish_reason) sem falhar em respostas incompletas."""
    choices = getattr(response, "choices", None) or []
    if not choices:
        return None, "", None
    message = getattr(choices[0], "message", None)
    return message, (getattr(message, "content", None) or ""), getattr(choices[0], "finish_reason", None)


def validate_api(api: dict) -> None:
    if not 1 <= int(api["concurrency"]) <= 200:
        raise ValueError("api.concurrency deve estar entre 1 e 200")
    if not 0 < float(api["start_rps"]) <= float(api["target_rps"]) <= 75:
        raise ValueError("Use 0 < api.start_rps <= api.target_rps <= 75")
    if int(api["max_attempts"]) < 1 or float(api["timeout_seconds"]) <= 0:
        raise ValueError("api.max_attempts e api.timeout_seconds devem ser positivos")


def require_token(api: dict) -> str:
    name = str(api.get("token_env") or "DEEPINFRA_TOKEN")
    token = os.environ.get(name, "").strip()
    if not token:
        raise EnvironmentError(f"Defina {name} como variável de ambiente para requisições novas.")
    return token


class AdaptiveRate:
    """Limita inícios/s; sobe após sucessos seguidos e cai pela metade ao receber 429."""

    def __init__(self, start: float, target: float, step: float, every: int):
        self.rate = float(start)
        self.max_rate = float(target)
        self.step = float(step)
        self.every = max(1, int(every))
        self.streak = 0
        self.next_start = 0.0
        self.lock = asyncio.Lock()

    @classmethod
    def from_api(cls, api: dict) -> "AdaptiveRate":
        return cls(api["start_rps"], api["target_rps"], api["ramp_step_rps"],
                   api["ramp_every_successes"])

    async def wait_slot(self) -> None:
        # Reserva o horário sob o lock, mas dorme fora dele: um 429 reduz a
        # taxa imediatamente, sem esperar a fila de workers.
        async with self.lock:
            now = time.monotonic()
            start = max(self.next_start, now)
            self.next_start = start + 1.0 / self.rate
        await asyncio.sleep(max(0.0, start - now))

    async def success(self) -> None:
        async with self.lock:
            self.streak += 1
            if self.streak >= self.every:
                self.rate = min(self.max_rate, self.rate + self.step)
                self.streak = 0

    async def limited(self) -> None:
        async with self.lock:
            self.rate = max(min(1.0, self.max_rate), self.rate / 2)
            self.streak = 0
            self.next_start = max(self.next_start, time.monotonic() + 1.0)


@dataclass
class Request:
    key: str
    body: dict
    pricing: dict
    group: tuple                      # unidade de validação prévia e de bloqueio
    ledger: dict = field(default_factory=dict)  # colunas descritivas do costs.csv
    data: Any = None                  # dados próprios do experimento


@dataclass
class Outcome:
    request: Request
    attempts: int
    latency_seconds: float
    response: Any = None
    cost: dict = field(default_factory=lambda: dict(UNKNOWN_COST))
    error: BaseException | None = None
    http_status: int | None = None
    transient: bool = False

    @property
    def ok(self) -> bool:
        return self.response is not None

    @property
    def error_type(self) -> str:
        return type(self.error).__name__ if self.error else ""

    @property
    def error_text(self) -> str:
        return str(self.error) if self.error else ""


@dataclass
class RunResult:
    completed: int = 0
    blocked: dict = field(default_factory=dict)   # grupo -> motivo
    stopped: str = ""                             # motivo de interrupção geral


def _usage(response: Any) -> dict:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    try:
        return usage.model_dump(exclude_none=True)
    except Exception:  # noqa: BLE001 - uso ausente vira custo desconhecido
        return {}


def _classify(exc: BaseException) -> tuple[int | None, bool]:
    import openai

    if isinstance(exc, openai.APIStatusError):
        return exc.status_code, exc.status_code in TRANSIENT_STATUS
    if isinstance(exc, openai.APIConnectionError):  # inclui APITimeoutError
        return None, True
    return None, False


class Executor:
    """Executa requisições com um pool de workers e registra cada tentativa.

    ``finish(outcome)`` monta e grava a linha de resultado do experimento.
    ``blocks_group(row, outcome)`` decide, na primeira requisição de cada grupo,
    se o restante do grupo deve ser pulado (padrão: erro permanente).
    """

    def __init__(self, api: dict, token: str, record_cost: Callable[[dict], None],
                 label: str = "", log: Callable[[str], None] = print):
        validate_api(api)
        self.api = api
        self.token = token
        self.record_cost = record_cost
        self.label = label
        self.log = log

    async def run(self, requests: list[Request], finish: Callable[[Outcome], dict],
                  blocks_group: Callable[[dict, Outcome], bool] | None = None) -> RunResult:
        import openai

        blocks_group = blocks_group or (lambda row, outcome: not outcome.ok and not outcome.transient)
        result = RunResult()
        if not requests:
            return result
        self.rate = AdaptiveRate.from_api(self.api)
        self.stop = asyncio.Event()
        self.total = len(requests)
        client = openai.AsyncOpenAI(api_key=self.token, base_url=str(self.api["base_url"]),
                                    max_retries=0, timeout=float(self.api["timeout_seconds"]))
        try:
            # Primeiro pedido de cada grupo sozinho: falhas de configuração custam uma chamada.
            first, rest, seen = [], [], set()
            for request in requests:
                (rest if request.group in seen else first).append(request)
                seen.add(request.group)
            for request in first:
                if self.stop.is_set():
                    break
                outcome = await self._attempts(client, request, result)
                row = self._finish(finish, outcome, result)
                if blocks_group(row, outcome):
                    result.blocked[request.group] = row.get("status") or outcome.error_type
            queue: asyncio.Queue = asyncio.Queue()
            for request in rest:
                queue.put_nowait(request)
            workers = [asyncio.create_task(self._worker(client, queue, finish, result))
                       for _ in range(min(int(self.api["concurrency"]), max(1, len(rest))))]
            # Espera todos terminarem a requisição em voo antes de propagar um erro.
            errors = [item for item in await asyncio.gather(*workers, return_exceptions=True)
                      if isinstance(item, BaseException)]
            if errors:
                raise errors[0]
        finally:
            await client.close()
        return result

    async def _worker(self, client, queue: asyncio.Queue, finish, result: RunResult) -> None:
        while not self.stop.is_set():
            try:
                request = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            if request.group in result.blocked:
                continue
            outcome = await self._attempts(client, request, result)
            self._finish(finish, outcome, result)
            if outcome.http_status in STOP_GROUP_STATUS:
                result.blocked.setdefault(request.group, f"HTTP {outcome.http_status}")

    def _finish(self, finish, outcome: Outcome, result: RunResult) -> dict:
        try:
            row = finish(outcome)
        except Exception:
            # O custo já foi registrado; interrompe sem cancelar quem está em voo.
            self.stop.set()
            raise
        result.completed += 1
        if result.completed % 25 == 0 or result.completed == self.total:
            self.log(f"{self.label}: {result.completed}/{self.total} concluídas; "
                     f"limite atual {self.rate.rate:.1f} req/s")
        return row

    async def _attempts(self, client, request: Request, result: RunResult) -> Outcome:
        maximum = int(self.api["max_attempts"])
        for attempt in range(1, maximum + 1):
            await self.rate.wait_slot()
            started = time.monotonic()
            response = error = None
            try:
                response = await client.chat.completions.create(**request.body)
            except Exception as exc:  # noqa: BLE001 - toda tentativa vai ao livro-caixa
                error = exc
            latency = round(time.monotonic() - started, 3)
            ledger = {**request.ledger, "request_sha256": request.key,
                      "timestamp_utc": now_utc(), "attempt": attempt}
            if response is not None:
                usage = _usage(response)
                cost = cost_fields(usage, request.pricing) if usage else dict(UNKNOWN_COST)
                self.record_cost({**ledger, "status": "ok", **cost,
                                  "request_id": getattr(response, "id", "") or ""})
                await self.rate.success()
                return Outcome(request, attempt, latency, response=response, cost=cost)
            status, transient = _classify(error)
            self.record_cost({**ledger, "status": "error", "http_status": status or "",
                              "error_type": type(error).__name__, "cost_source": "unknown"})
            if status == 429:
                await self.rate.limited()
            if status in STOP_ALL_STATUS and not self.stop.is_set():
                result.stopped = f"HTTP {status}: {str(error)[:300]}"
                self.stop.set()
            if not transient or attempt == maximum or self.stop.is_set():
                return Outcome(request, attempt, latency, error=error,
                               http_status=status, transient=transient)
            await asyncio.sleep(min(2 ** attempt, 12) + random.random() * .25)
        raise AssertionError("inalcançável")
