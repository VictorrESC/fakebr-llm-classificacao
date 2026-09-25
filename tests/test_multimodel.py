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

    def test_pairs_from_and_length_baseline(self):
        with workspace() as ws:
            corpus = ws.corpus(pairs=6)
            # Verdadeiras longas: a regra de comprimento separa as classes.
            for index in range(1, 7):
                (ws.base / "data" / "corpus" / "full_texts" / "true" / f"{index}.txt").write_text(
                    "palavra " * (50 + index), encoding="utf-8")
            sizes = ("experiment.dev_pairs=1", "experiment.eval_pairs=2")
            ws.cli(*EXPERIMENT, "--run", "origem", "--stage", "prepare", *sizes, *corpus)
            ws.cli(*EXPERIMENT, "--run", "copia", "--stage", "prepare", *sizes, *corpus,
                   "experiment.seed=999", "experiment.pairs_from=origem")

            def pairs(run):
                return [(r["stage"], r["pair_id"]) for r in read_csv(ws.run_dir(run) / "manifest.csv")]
            self.assertEqual(pairs("copia"), pairs("origem"))
            info = json.loads((ws.run_dir("copia") / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(info["pairs_from"]["run"], "origem")
            with self.assertRaisesRegex(ValueError, "Pares fixados: dev=1, eval=2"):
                ws.cli(*EXPERIMENT, "--run", "maior", "--stage", "prepare", *corpus,
                       "experiment.dev_pairs=1", "experiment.eval_pairs=3",
                       "experiment.pairs_from=origem")
            with self.assertRaisesRegex(FileNotFoundError, "pairs_from"):
                ws.cli(*EXPERIMENT, "--run", "orfao", "--stage", "prepare", *sizes, *corpus,
                       "experiment.pairs_from=inexistente")

            ws.cli(*EXPERIMENT, "--run", "copia", "--stage", "eval", ONLY_T0)
            ws.cli(*REPORT, "--run", "copia", "--stage", "eval", ONLY_T0)
            [baseline] = read_csv(ws.run_dir("copia") / "report_eval" / "length_baseline.csv")
            # Ajuste só nos 3 pares fora da amostra; o eval (2 pares) não entra.
            self.assertEqual(baseline["fit_pairs"], "3")
            self.assertEqual(baseline["above_label"], "VERDADEIRA")
            self.assertEqual(float(baseline["accuracy"]), 1.0)
            self.assertNotEqual(baseline["macro_f1_ci95_low"], "")

    def test_length_rule(self):
        from fakebr.metrics import apply_length_rule, fit_length_rule

        labels = ("FALSA", "VERDADEIRA")
        train = [(10, "FALSA"), (20, "FALSA"), (30, "VERDADEIRA"), (40, "VERDADEIRA")]
        rule = fit_length_rule(train, labels)
        self.assertEqual((rule["threshold_words"], rule["above_label"], rule["fit_accuracy"]),
                         (20, "VERDADEIRA", 1.0))
        self.assertEqual(apply_length_rule(rule, 25), "VERDADEIRA")
        self.assertEqual(apply_length_rule(rule, 20), "FALSA")
        # Sentido invertido: textos longos são os falsos.
        flipped = fit_length_rule([(w, labels[1 - labels.index(g)]) for w, g in train], labels)
        self.assertEqual((flipped["threshold_words"], flipped["above_label"]), (20, "FALSA"))

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
