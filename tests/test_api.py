"""Executor de requisições: erros, bloqueios, livro-caixa e limite de taxa."""

import asyncio
import time
import unittest
from unittest.mock import patch

from fakes import FakeClient, completion, status_error

from fakebr.api import AdaptiveRate, Executor, Request

API = {"base_url": "https://example.invalid", "start_rps": 75.0, "target_rps": 75.0,
       "ramp_step_rps": 5.0, "ramp_every_successes": 20, "concurrency": 4,
       "timeout_seconds": 5, "max_attempts": 3}
PRICING = {"input_usd_per_million": 1.0, "output_usd_per_million": 1.0,
           "cached_usd_per_million": None}


def requests(groups=("m1", "m2"), per_group=4):
    return [Request(key=f"{group}-{i}", body={"model": group, "i": i}, pricing=PRICING,
                    group=(group,), ledger={"model": group})
            for group in groups for i in range(per_group)]


def execute(items, handler, **kwargs):
    FakeClient.reset(handler)
    ledger, rows = [], []

    def finish(outcome):
        row = {"key": outcome.request.key, "ok": outcome.ok, "http": outcome.http_status,
               "status": "ok" if outcome.ok else "error"}
        rows.append(row)
        return row

    with patch("openai.AsyncOpenAI", FakeClient):
        result = asyncio.run(Executor({**API, **kwargs}, "token", ledger.append,
                                      log=lambda _: None).run(items, finish))
    return result, rows, ledger


class ExecutorTest(unittest.TestCase):
    def test_request_error_does_not_stop_others(self):
        async def handler(kwargs):
            if kwargs["i"] == 2:
                raise status_error(400)
            return completion(kwargs)
        result, rows, ledger = execute(requests(), handler)
        self.assertEqual(len(rows), 8)
        self.assertEqual(sum(not r["ok"] for r in rows), 2)
        self.assertEqual(result.blocked, {})
        self.assertEqual(len(ledger), 8)  # 400 não é repetido

    def test_unexpected_exception_is_recorded(self):
        async def handler(kwargs):
            if kwargs["i"] == 1:
                raise ValueError("resposta malformada")
            return completion(kwargs)
        _, rows, ledger = execute(requests(), handler)
        errors = [r for r in ledger if r["status"] == "error"]
        self.assertEqual({r["error_type"] for r in errors}, {"ValueError"})
        self.assertTrue(all(r["cost_source"] == "unknown" for r in errors))
        self.assertEqual(len(rows), 8)

    def test_preflight_failure_blocks_only_its_group(self):
        async def handler(kwargs):
            if kwargs["model"] == "m1":
                raise status_error(404)
            return completion(kwargs)
        result, rows, _ = execute(requests(), handler)
        self.assertEqual(set(result.blocked), {("m1",)})
        self.assertEqual(sum(r["key"].startswith("m1") for r in rows), 1)
        self.assertEqual(sum(r["key"].startswith("m2") for r in rows), 4)

    def test_auth_error_stops_and_waits_for_in_flight(self):
        async def handler(kwargs):
            if kwargs["model"] == "m2" and kwargs["i"] == 3:
                raise status_error(401)
            await asyncio.sleep(.05)
            return completion(kwargs)
        items = requests(per_group=20)
        result, rows, ledger = execute(items, handler)
        self.assertIn("401", result.stopped)
        self.assertLess(len(rows), len(items))
        # Toda chamada feita tem resultado e linha no livro-caixa (nada cancelado).
        self.assertEqual(len(rows), len(FakeClient.calls))
        self.assertEqual(len(ledger), len(FakeClient.calls))

    def test_transient_error_is_retried_and_rate_drops(self):
        async def handler(kwargs):
            if sum(1 for c in FakeClient.calls if c["i"] == 0 and c["model"] == "m1") == 1 \
                    and kwargs["i"] == 0 and kwargs["model"] == "m1":
                raise status_error(429)
            return completion(kwargs)
        with patch("fakebr.api.random.random", return_value=0):
            result, rows, ledger = execute(requests(("m1",), 1), handler)
        self.assertEqual([r["status"] for r in ledger], ["error", "ok"])
        self.assertEqual(ledger[0]["http_status"], 429)
        self.assertTrue(rows[0]["ok"])

    def test_rate_limit_is_not_blocked_by_waiting_workers(self):
        async def scenario():
            rate = AdaptiveRate(0.5, 0.5, 0, 1)
            waiters = [asyncio.create_task(rate.wait_slot()) for _ in range(3)]
            await asyncio.sleep(.01)
            started = time.monotonic()
            await asyncio.wait_for(rate.limited(), timeout=.5)
            elapsed = time.monotonic() - started
            for task in waiters:
                task.cancel()
            return elapsed, rate.rate
        elapsed, rate = asyncio.run(scenario())
        self.assertLess(elapsed, .5)
        self.assertEqual(rate, .5)


if __name__ == "__main__":
    unittest.main()
