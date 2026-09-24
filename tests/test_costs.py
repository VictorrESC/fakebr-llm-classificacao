"""Livro-caixa central: sem duplicatas, inclusive depois de mover o projeto."""

import tempfile
import unittest
from pathlib import Path

from fakebr import costs
from fakebr.io import append_csv, read_csv

ERROR_ROW = {"timestamp_utc": "2026-09-24T10:00:00+00:00", "stage": "dev", "model": "m",
             "request_sha256": "a" * 64, "attempt": 1, "status": "error",
             "error_type": "APITimeoutError", "cost_source": "unknown"}
OK_ROW = {**ERROR_ROW, "attempt": 2, "status": "ok", "request_id": "req-1",
          "provider_cost_usd": "0.00001", "cost_source": "provider"}


class CostLedgerTest(unittest.TestCase):
    def test_ids_do_not_depend_on_absolute_path(self):
        self.assertEqual(costs._ledger_id(Path("C:/a/runs/piloto"), ERROR_ROW),
                         costs._ledger_id(Path("/outro/lugar/runs/piloto"), ERROR_ROW))

    def test_sync_is_idempotent_after_moving(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            run = base / "antes" / "runs" / "piloto"
            run.mkdir(parents=True)
            for row in (ERROR_ROW, OK_ROW):
                append_csv(run / "costs.csv", row, list(OK_ROW))
            ledger = base / "custos_totais.csv"
            self.assertEqual(costs.sync_project(base / "antes" / "runs", ledger), 2)
            (base / "antes").rename(base / "depois")
            self.assertEqual(costs.sync_project(base / "depois" / "runs", ledger), 0)
            rows = read_csv(ledger)
            self.assertEqual([r["accounted_cost_usd"] for r in rows], ["", "0.00001"])
            self.assertEqual({r["source_type"] for r in rows}, {"pilot"})

    def test_append_keeps_existing_header(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "x.csv"
            append_csv(path, {"a": 1, "b": 2}, ["a", "b"])
            append_csv(path, {"b": 3, "a": 4}, ["b", "a"])
            self.assertEqual(read_csv(path), [{"a": "1", "b": "2"}, {"a": "4", "b": "3"}])
            with self.assertRaises(RuntimeError):
                append_csv(path, {"a": 1, "c": 2}, ["a", "c"])


if __name__ == "__main__":
    unittest.main()
