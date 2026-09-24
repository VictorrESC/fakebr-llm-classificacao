"""Sondas de viabilidade com cliente simulado; nenhuma chamada paga."""

import io
import shutil
import unittest
from contextlib import redirect_stdout

from fakes import FakeClient, completion, workspace

from fakebr.io import read_csv

EXPERIMENT = ("experiment", "viabilidade")


class ViabilityTest(unittest.TestCase):
    def test_run_resume_logprobs_reuse_and_guard(self):
        with workspace() as ws:
            ws.cli(*EXPERIMENT, "--run", "viab", "experiment.include_logprobs_probe=true")
            ws.cli(*EXPERIMENT, "--run", "viab")
            out = ws.run_dir("viab")
            self.assertEqual(len(FakeClient.calls), 18)
            self.assertEqual(len(read_csv(out / "results.csv")), 18)
            self.assertEqual(len(read_csv(out / "costs.csv")), 18)
            self.assertEqual(len(read_csv(ws.base / "runs" / "custos_totais.csv")), 18)
            summary = read_csv(out / "summary.csv")
            self.assertEqual(len(summary), 3)
            self.assertTrue(all(r["base_approved"] == "True" and r["logprobs_supported"] == "1"
                                for r in summary))

            ws.cli(*EXPERIMENT, "--run", "reuso", "experiment.include_logprobs_probe=true",
                   token=None)
            self.assertEqual(len(FakeClient.calls), 18)
            self.assertEqual(len(read_csv(ws.run_dir("reuso") / "results.csv")), 18)
            self.assertEqual(read_csv(ws.run_dir("reuso") / "costs.csv"), [])

            with self.assertRaisesRegex(ValueError, "congelado"):
                ws.cli(*EXPERIMENT, "--run", "viab", "experiment.max_tokens=20")
            # Ritmo da API pode mudar ao retomar; relatório não chama a API.
            ws.cli(*EXPERIMENT, "--run", "viab", "api.concurrency=1", token=None)
            ws.cli("report", "viabilidade", "--run", "viab", token=None)
            self.assertEqual(len(FakeClient.calls), 18)

            # Subcomandos auxiliares com o mesmo projeto temporário.
            shutil.rmtree(ws.base / "cache")
            output = io.StringIO()
            with redirect_stdout(output):
                ws.cli("cache-import", "--run", "viab", token=None)
                ws.cli("costs", token=None)
                ws.cli("status", token=None)
            text = output.getvalue()
            self.assertIn("18 respostas importadas", text)
            self.assertIn("Novas tentativas importadas: 0", text)
            self.assertRegex(text, r"viab\s+viabilidade")

    def test_model_mismatch_blocks_followup(self):
        with workspace() as ws:
            async def mismatch(kwargs):
                wrong = "unexpected/model" if kwargs["model"].startswith("deepseek-ai/") else None
                return completion(kwargs, model=wrong)
            FakeClient.reset(mismatch)
            ws.cli(*EXPERIMENT, "--run", "viab")
            rows = read_csv(ws.run_dir("viab") / "results.csv")
            self.assertEqual(len(rows), 11)  # 1 DeepSeek bloqueada + 5 Qwen + 5 Gemma
            self.assertEqual(sum(r["status"] == "model_mismatch" for r in rows), 1)
            ws.cli(*EXPERIMENT, "--run", "viab")
            self.assertEqual(len(FakeClient.calls), 11)
            summary = {r["model"]: r for r in read_csv(ws.run_dir("viab") / "summary.csv")}
            self.assertEqual(summary["deepseek-ai/DeepSeek-V4.1-Flash"]["base_approved"], "False")

    def test_reasoning_and_missing_usage_are_flagged(self):
        with workspace() as ws:
            async def leaky(kwargs):
                if kwargs["model"].startswith("Qwen/"):
                    return completion(kwargs, content="<think>x</think>FALSA")
                return completion(kwargs, usage=None)
            FakeClient.reset(leaky)
            ws.cli(*EXPERIMENT, "--run", "viab")
            summary = {r["model"]: r for r in read_csv(ws.run_dir("viab") / "summary.csv")}
            self.assertEqual(summary["Qwen/Qwen3.8-27B"]["reasoning_present"], "5")
            self.assertEqual(summary["google/gemma-4-31B-it"]["usage_missing"], "5")
            self.assertTrue(all(r["base_approved"] == "False" for r in summary.values()))


if __name__ == "__main__":
    unittest.main()
