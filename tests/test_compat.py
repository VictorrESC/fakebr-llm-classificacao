"""Compatibilidade com runs e cache existentes.

golden_keys.json foi gerado pelo código anterior à reorganização. Se estes
testes falharem, as chaves das requisições mudaram: o cache e os runs antigos
deixariam de ser reconhecidos.
"""

import json
import tempfile
import unittest
from pathlib import Path

from fakebr import legacy, project
from fakebr.api import request_payload
from fakebr.experiments import multimodel, viabilidade
from fakebr.experiments.common import prompt_set

GOLDEN = json.loads((Path(__file__).parent / "golden_keys.json").read_text(encoding="utf-8"))
TEXT = "Texto {com} chaves\nlinha 2."


class CompatibilityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = project.project_root()
        cls.cfg = project.compose(cls.root, "fakebr_multimodel", [])
        cls.experiment = multimodel.freeze(cls.cfg, cls.root)

    def test_prompt_texts_unchanged(self):
        prompts = prompt_set(self.experiment)
        for prompt_id, digest in GOLDEN["prompt_sha256"].items():
            self.assertEqual(prompts.fingerprint(prompt_id), digest, prompt_id)

    def test_request_keys_unchanged(self):
        prompts = prompt_set(self.experiment)
        models = {m["id"]: m for m in self.experiment["models"]}
        for item in GOLDEN["keys"]:
            body, key = request_payload(
                base_url=self.cfg["api"]["base_url"], model=models[item["model"]],
                temperature=item["temperature"], top_p=self.experiment["top_p"],
                max_tokens=self.experiment["max_tokens"], enable_prompt_cache_key=True,
                prompt=prompts.render(item["prompt"], TEXT),
                shared_prefix=prompts.shared_prefix(item["prompt"]),
                repetition=item["repetition"])
            self.assertEqual(key, item["key"])
            self.assertEqual(body["extra_body"]["prompt_cache_key"], item["cache_key"])

    def test_viability_keys_unchanged(self):
        cfg = project.compose(self.root, "viabilidade", ["experiment.include_logprobs_probe=true"])
        experiment = viabilidade.freeze(cfg, self.root)
        self.assertEqual([p["key"] for p in viabilidade.plans(experiment, cfg["api"])],
                         GOLDEN["viability_keys"])

    def test_label_keys_are_folder_names(self):
        self.assertEqual(list(self.experiment["prompt_set"]["labels"]), ["fake", "true"])
        with self.assertRaisesRegex(ValueError, "aspas"):
            prompt_set({"prompt_set": {**self.experiment["prompt_set"],
                                       "labels": {"fake": "FALSA", True: "VERDADEIRA"}}})

    def test_frozen_yaml_round_trip(self):
        text = project.dump_yaml({"experiment": self.experiment})
        self.assertEqual(project.yaml.safe_load(text)["experiment"], self.experiment)

    def legacy_models(self):
        return [{k: v for k, v in m.items() if k != "name"} for m in self.experiment["models"]]

    def test_legacy_pilot_conversion(self):
        prompts = prompt_set(self.experiment)
        config = {
            "created_at": "2026-09-23T20:27:49+00:00",
            "dataset_root": "V:\\Dev\\projeto\\data\\raw\\Fake.br-Corpus",
            "dataset_git_revision": "780f5516", "seed": 1, "dev_pairs": 10, "eval_pairs": 200,
            "request_settings": {
                "base_url": self.cfg["api"]["base_url"], "models": self.legacy_models(),
                "conditions": self.experiment["conditions"], "top_p": 1.0, "max_tokens": 16,
                "n": 1, "stream": False, "enable_prompt_cache_key": True,
                "prompt_sha256": dict(GOLDEN["prompt_sha256"])},
            "prompt_introductions": prompts.introductions,
        }
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            (out / "config.json").write_text(json.dumps(config), encoding="utf-8")
            frozen = legacy.convert(out, self.root)
            self.assertEqual(frozen["experiment"]["dataset"]["dir"], "Fake.br-Corpus")
            self.assertEqual(frozen["experiment"]["prompts"], ["A"])
            self.assertEqual(frozen["experiment"]["prompt_set"], self.experiment["prompt_set"])
            config["request_settings"]["prompt_sha256"]["B"] = "0" * 64
            (out / "config.json").write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Prompt B"):
                legacy.convert(out, self.root)

    def test_legacy_viability_conversion(self):
        config = {
            "version": 1, "base_url": self.cfg["api"]["base_url"], "planned_requests": 18,
            "request_sha256": GOLDEN["viability_keys"], "payloads_sha256": "x",
            "settings": {"experiment": {"models": self.legacy_models(), "top_p": 1.0,
                                        "max_tokens": 16},
                         "api": {"base_url": self.cfg["api"]["base_url"],
                                 "enable_prompt_cache_key": True},
                         "viability": {"include_logprobs_probe": True}},
            "control_sha256": viabilidade.controls_fingerprint(),
        }
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            (out / "config.json").write_text(json.dumps(config), encoding="utf-8")
            self.assertEqual(legacy.convert(out, self.root)["runner"], "viabilidade")
            config["settings"]["experiment"]["max_tokens"] = 20
            (out / "config.json").write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "não batem"):
                legacy.convert(out, self.root)


if __name__ == "__main__":
    unittest.main()
