"""Cliente OpenAI simulado e área de trabalho temporária: nenhum teste chama a API."""

from __future__ import annotations

import os
import shutil
import tempfile
import types
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import httpx
import openai

from fakebr import cli


class Usage:
    def __init__(self, data: dict):
        self.data = data

    def model_dump(self, **kwargs):
        return dict(self.data)


DEFAULT_USAGE = {"prompt_tokens": 90, "completion_tokens": 2,
                 "prompt_tokens_details": {"cached_tokens": 7}, "estimated_cost": 0.00001}


def completion(kwargs: dict, content: str = "FALSA", *, model: str | None = None,
               usage: dict | None = DEFAULT_USAGE, finish_reason: str = "stop",
               choices: bool = True, reasoning: str | None = None):
    logprobs = (types.SimpleNamespace(content=[types.SimpleNamespace(top_logprobs=[
        types.SimpleNamespace(token=" FALSA", logprob=-0.2)])]) if kwargs.get("logprobs") else None)
    FakeClient.counter += 1
    return types.SimpleNamespace(
        model=model or kwargs["model"], id=f"fake-{FakeClient.counter}",
        usage=Usage(usage) if usage is not None else None,
        choices=[types.SimpleNamespace(
            message=types.SimpleNamespace(content=content, reasoning_content=reasoning),
            finish_reason=finish_reason, logprobs=logprobs)] if choices else [],
    )


def status_error(code: int) -> openai.APIStatusError:
    request = httpx.Request("POST", "https://example.invalid/v1/chat/completions")
    classes = {400: openai.BadRequestError, 401: openai.AuthenticationError,
               404: openai.NotFoundError, 429: openai.RateLimitError,
               500: openai.InternalServerError}
    return classes[code](f"HTTP {code}", response=httpx.Response(code, request=request), body=None)


class FakeClient:
    """Substitui openai.AsyncOpenAI; ``handler(kwargs)`` decide cada resposta."""

    calls: list[dict] = []
    counter = 0
    handler = None

    def __init__(self, **kwargs):
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self._create))

    async def _create(self, **kwargs):
        FakeClient.calls.append(kwargs)
        return await FakeClient.handler(kwargs)

    async def close(self):
        pass

    @classmethod
    def reset(cls, handler=None):
        async def default(kwargs):
            return completion(kwargs)
        cls.calls, cls.counter, cls.handler = [], 0, handler or default


class Workspace:
    """Projeto temporário: runs/, cache/, livro-caixa e corpus fora da pasta real."""

    def __init__(self, base: Path):
        self.base = base
        self.paths = [f"paths.runs='{(base / 'runs').as_posix()}'",
                      f"paths.cache='{(base / 'cache').as_posix()}'",
                      f"paths.ledger='{(base / 'runs' / 'custos_totais.csv').as_posix()}'",
                      f"paths.data='{(base / 'data').as_posix()}'"]

    def corpus(self, pairs: int = 2, name: str = "corpus") -> list[str]:
        for label, prefix in (("fake", "Texto falso"), ("true", "Texto verdadeiro")):
            folder = self.base / "data" / name / "full_texts" / label
            folder.mkdir(parents=True, exist_ok=True)
            for index in range(1, pairs + 1):
                (folder / f"{index}.txt").write_text(f"{prefix} número {index}.\r\nSegunda linha.",
                                                     encoding="utf-8")
        return [f"datasets.fakebr.dir={name}", "datasets.fakebr.fingerprint.full_texts=null"]

    def cli(self, *argv: str, token: str | None = "test-sentinel") -> None:
        with patch("openai.AsyncOpenAI", FakeClient), \
                patch.dict(os.environ, {"DEEPINFRA_TOKEN": token or ""}):
            cli.run_cli([*argv, *self.paths])

    def run_dir(self, name: str) -> Path:
        return self.base / "runs" / name


@contextmanager
def workspace():
    base = Path(tempfile.mkdtemp(prefix="fakebr-test-"))
    try:
        FakeClient.reset()
        yield Workspace(base)
    finally:
        shutil.rmtree(base, ignore_errors=True)
