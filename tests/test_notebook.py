"""Executa notebooks/resultados.ipynb num kernel real, sobre runs simulados."""

import sys
import unittest
from pathlib import Path

from fakes import workspace

try:
    import nbformat
    from nbclient import NotebookClient
except ImportError:  # pragma: no cover - ambiente sem as dependências de notebook
    nbformat = None

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "resultados.ipynb"


def run_notebook(directory: Path, **parameters) -> nbformat.NotebookNode:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    cell = next(c for c in notebook.cells if "parameters" in c.metadata.get("tags", []))
    cell.source = "\n".join(f"{name} = {value!r}" for name, value in parameters.items())
    notebook.cells.insert(0, nbformat.v4.new_code_cell("import sys; print(sys.executable)"))
    NotebookClient(notebook, timeout=600, kernel_name="python3",
                   resources={"metadata": {"path": str(directory)}}).execute()
    return notebook


@unittest.skipIf(nbformat is None, "nbclient/nbformat não instalados")
class NotebookTest(unittest.TestCase):
    def test_notebook_reports_png_and_pdf(self):
        with workspace() as ws:
            ws.cli("experiment", "fakebr_multimodel", "--run", "piloto", "--stage", "prepare",
                   "experiment.dev_pairs=1", "experiment.eval_pairs=1", *ws.corpus())
            ws.cli("experiment", "fakebr_multimodel", "--run", "piloto", "--stage", "dev")
            ws.cli("experiment", "viabilidade", "--run", "viab")

            common = {"FORMATOS": ["png", "pdf"], "EXECUTAR_ETAPAS": [], "OVERRIDES_CRIACAO": []}
            notebook = run_notebook(ws.base, EXPERIMENTO="fakebr_multimodel", RUN="piloto",
                                    ETAPA="dev", RECORTE=ws.paths, **common)
            kernel_python = notebook.cells[0].outputs[0]["text"].strip()
            self.assertEqual(Path(kernel_python).resolve(), Path(sys.executable).resolve())
            report = ws.run_dir("piloto") / "report_dev"
            for stem in ("macro_f1", "per_class", "confusion", "stability", "cost_by_condition"):
                for suffix in ("png", "pdf"):
                    self.assertGreater((report / f"{stem}.{suffix}").stat().st_size, 0)
            self.assertTrue((report / "macro_f1.pdf").read_bytes().startswith(b"%PDF"))

            # PDFs sem data embutida: o mesmo relatório gera os mesmos bytes.
            first = (report / "macro_f1.pdf").read_bytes()
            ws.cli("report", "fakebr_multimodel", "--run", "piloto", "report.formats=[pdf]")
            self.assertEqual((report / "macro_f1.pdf").read_bytes(), first)
            self.assertFalse((report / "macro_f1.png").exists())

            run_notebook(ws.base, EXPERIMENTO="viabilidade", RUN="viab", ETAPA=None,
                         RECORTE=ws.paths, **common)
            for stem in ("viability_status", "viability_cost"):
                for suffix in ("png", "pdf"):
                    self.assertTrue((ws.run_dir("viab") / "report" / f"{stem}.{suffix}").is_file())


if __name__ == "__main__":
    unittest.main()
