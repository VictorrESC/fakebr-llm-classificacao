"""Fluxo completo do experimento multimodelo com cliente simulado."""

import json
import shutil
import unittest

from fakes import FakeClient, completion, workspace

from fakebr.io import read_csv

EXPERIMENT = ("experiment", "fakebr_multimodel")
REPORT = ("report", "fakebr_multimodel")
ONLY_T0 = "select.temperatures=[0.0]"


class MultimodelTest(unittest.TestCase):
    def prepare(self, ws, run="piloto", *extra):
        corpus = ws.corpus()
        ws.cli(*EXPERIMENT, "--run", run, "--stage", "prepare", "experiment.dev_pairs=1",
               "experiment.eval_pairs=1", *corpus, *extra)
        return ws.run_dir(run)

    def test_prepare_run_resume_report_reuse_eval(self):
        with workspace() as ws:
            out = self.prepare(ws)
            self.assertTrue((out / "experiment.yaml").is_file())
            self.assertTrue((out / "provenance.json").is_file())
            manifest = read_csv(out / "manifest.csv")
            self.assertEqual([r["stage"] for r in manifest], ["dev", "dev", "eval", "eval"])

            ws.cli(*EXPERIMENT, "--run", "piloto", "--stage", "dev", ONLY_T0)
            ws.cli(*EXPERIMENT, "--run", "piloto", "--stage", "dev", ONLY_T0)
            self.assertEqual(len(FakeClient.calls), 6)
            rows = read_csv(out / "responses.csv")
            self.assertEqual(len(rows), 6)
            self.assertEqual(len({r["prompt_cache_key"] for r in rows}), 3)
            ledger = read_csv(out / "costs.csv")
            self.assertAlmostEqual(sum(float(r["provider_cost_usd"]) for r in ledger), .00006)
            self.assertEqual(len(read_csv(ws.base / "runs" / "custos_totais.csv")), 6)

            ws.cli(*REPORT, "--run", "piloto", ONLY_T0)
            scores = json.loads((out / "report_dev" / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(len(scores), 3)
            self.assertTrue(all(abs(v["macro_f1"] - 1 / 3) < 1e-9 for v in scores.values()))

            ws.cli(*EXPERIMENT, "--run", "piloto", "--stage", "dev")
            ws.cli(*REPORT, "--run", "piloto", "--stage", "dev")
            self.assertEqual(len(read_csv(out / "responses.csv")), 42)
            self.assertEqual(len(read_csv(out / "report_dev" / "metrics.csv")), 21)
            repeats = read_csv(out / "report_dev" / "repeat_summary.csv")
            self.assertEqual(len(repeats), 9)
            self.assertEqual({r["disagreement_fraction"] for r in repeats
                              if r["repetitions"] == "3"}, {"0.0"})
            for name in ("macro_f1.png", "per_class.png", "confusion.png",
                         "stability.png", "cost_by_condition.png"):
                self.assertGreater((out / "report_dev" / name).stat().st_size, 0)

            # Novo run com as mesmas requisições: tudo do cache, sem token nem custo.
            calls = len(FakeClient.calls)
            ws.cli(*EXPERIMENT, "--run", "reuso", "--stage", "prepare", "experiment.dev_pairs=1",
                   "experiment.eval_pairs=1", *ws.corpus())
            ws.cli(*EXPERIMENT, "--run", "reuso", "--stage", "dev", token=None)
            self.assertEqual(len(FakeClient.calls), calls)
            self.assertEqual(len(read_csv(ws.run_dir("reuso") / "responses.csv")), 42)
            self.assertEqual(read_csv(ws.run_dir("reuso") / "costs.csv"), [])

            ws.cli(*EXPERIMENT, "--run", "piloto", "--stage", "eval", ONLY_T0)
            ws.cli(*REPORT, "--run", "piloto", "--stage", "eval", ONLY_T0)
            metrics = read_csv(out / "report_eval" / "metrics.csv")
            self.assertEqual(len(metrics), 3)
            self.assertTrue(all(r["macro_f1_ci95_low"] != "" for r in metrics))
            self.assertEqual(len(read_csv(ws.base / "runs" / "custos_totais.csv")), 48)
            executions = (out / "executions.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(executions), 8)

    def test_frozen_plan_survives_conf_changes(self):
        with workspace() as ws:
            # O run nasce com um único modelo; retomadas sem override seguem o congelado.
            self.prepare(ws, "piloto", "experiment.models=[qwen38_27b]")
            ws.cli(*EXPERIMENT, "--run", "piloto", "--stage", "dev", ONLY_T0)
            self.assertEqual({c["model"] for c in FakeClient.calls}, {"Qwen/Qwen3.8-27B"})
            with self.assertRaisesRegex(ValueError, "congelado"):
                ws.cli(*EXPERIMENT, "--run", "piloto", "--stage", "dev", "experiment.max_tokens=20")
            with self.assertRaisesRegex(ValueError, "congelado"):
                ws.cli(*REPORT, "--run", "piloto", "experiment.prompts=[A,B]")
            with self.assertRaisesRegex(ValueError, "pertence"):
                ws.cli("report", "viabilidade", "--run", "piloto")
            with self.assertRaisesRegex(FileExistsError, "já foi preparado"):
                ws.cli(*EXPERIMENT, "--run", "piloto", "--stage", "prepare")

    def test_project_can_move(self):
        with workspace() as ws:
            self.prepare(ws)
            # Corpus e runs em outro lugar: nada guarda caminho absoluto.
            shutil.move(ws.base / "data", ws.base / "novo_data")
            shutil.move(ws.base / "runs", ws.base / "novo_runs")
            ws.paths = [p.replace("/data'", "/novo_data'").replace("/runs'", "/novo_runs'")
                        .replace("/runs/custos", "/novo_runs/custos") for p in ws.paths]
            ws.cli(*EXPERIMENT, "--run", "piloto", "--stage", "dev", ONLY_T0)
            ws.cli(*REPORT, "--run", "piloto", ONLY_T0)
            self.assertTrue((ws.base / "novo_runs" / "piloto" / "report_dev" / "metrics.csv").exists())

    def test_failed_first_stage_leaves_no_run(self):
        with workspace() as ws:
            with self.assertRaisesRegex(FileNotFoundError, "Dataset não encontrado"):
                ws.cli(*EXPERIMENT, "--run", "piloto", "--stage", "prepare",
                       "datasets.fakebr.dir=inexistente")
            self.assertFalse(ws.run_dir("piloto").exists())
            with self.assertRaisesRegex(ValueError, "não existe.*--stage prepare"):
                ws.cli(*EXPERIMENT, "--run", "outro", "--stage", "dev")

    def test_empty_choices_and_missing_responses(self):
        with workspace() as ws:
            out = self.prepare(ws)

            async def empty_first(kwargs):
                return completion(kwargs, choices=len(FakeClient.calls) > 1)
            FakeClient.reset(empty_first)
            ws.cli(*EXPERIMENT, "--run", "piloto", "--stage", "dev", ONLY_T0)
            rows = read_csv(out / "responses.csv")
            self.assertEqual(sum(r["status"] == "technical_error" for r in rows), 1)
            # A resposta vazia foi cobrada: aparece no livro-caixa como tentativa ok.
            self.assertEqual(len(read_csv(out / "costs.csv")), 6)
            with self.assertRaisesRegex(RuntimeError, "faltam 1 respostas"):
                ws.cli(*REPORT, "--run", "piloto", ONLY_T0)
            ws.cli(*EXPERIMENT, "--run", "piloto", "--stage", "dev", ONLY_T0)
            self.assertEqual(len(FakeClient.calls), 7)
            ws.cli(*REPORT, "--run", "piloto", ONLY_T0)


if __name__ == "__main__":
    unittest.main()
