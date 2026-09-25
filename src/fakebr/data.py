"""Corpus Fake.Br: leitura, sorteio do manifesto, impressão digital e download."""

from __future__ import annotations

import json
import random
import shutil
import statistics
import subprocess
import tempfile
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path, PurePosixPath

from .io import canonical_hash, now_utc, sha256


SOURCE_FILE = "SOURCE.json"


def read_text(path: Path) -> str:
    # Leitura em modo texto normaliza CRLF -> LF: o hash não depende do SO.
    # Arquivos com codificação ilegível exigem inspeção, não correção silenciosa.
    return path.read_text(encoding="utf-8-sig").strip()


def files_by_id(folder: Path) -> dict[str, Path]:
    if not folder.is_dir():
        raise FileNotFoundError(f"Pasta ausente: {folder}")
    files = sorted(folder.glob("*.txt"))
    if not files:
        raise ValueError(f"Nenhum .txt encontrado em {folder}")
    result = {path.stem: path for path in files}
    if len(result) != len(files):
        raise ValueError(f"IDs repetidos em {folder}")
    return result


def id_order(identifier: str) -> tuple[int, int | str]:
    return (0, int(identifier)) if identifier.isdecimal() else (1, identifier)


def dataset_revision(root: Path) -> str | None:
    """Commit git do corpus clonado, ou a revisão registrada por fetch-data."""
    try:
        result = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                                capture_output=True, text=True, check=False)
        if result.returncode == 0 and (root / ".git").exists():
            return result.stdout.strip()
    except FileNotFoundError:
        pass
    source = root / SOURCE_FILE
    if source.exists():
        return json.loads(source.read_text(encoding="utf-8")).get("revision")
    return None


def length_summary(values: list[int]) -> dict:
    ordered = sorted(values)
    return {
        "n": len(ordered), "min": ordered[0], "median": statistics.median(ordered),
        "mean": round(statistics.mean(ordered), 2),
        "p90": ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))],
        "max": ordered[-1],
    }


def fingerprint(entries: list[tuple[str, str]]) -> str:
    """Hash de (caminho relativo, SHA-256 do texto) de todos os arquivos lidos."""
    return canonical_hash(sorted(entries))


def word_counts(dataset: Path, texts: str, labels: dict[str, str]) -> dict[str, dict[str, int]]:
    """Palavras de cada texto dos pares completos: ``{pair_id: {rótulo: palavras}}``.

    Conta como o manifesto (``len(texto.split())``).
    """
    folders = {cls: files_by_id(dataset / texts / cls) for cls in labels}
    common = set.intersection(*(set(files) for files in folders.values()))
    return {pair_id: {label: len(read_text(folders[cls][pair_id]).split())
                      for cls, label in labels.items()}
            for pair_id in sorted(common, key=id_order)}


def build_manifest(dataset: Path, texts: str, labels: dict[str, str], seed: int,
                   dev_pairs: int, eval_pairs: int,
                   expected_fingerprint: str | None = None,
                   pairs: dict[str, list[str]] | None = None) -> tuple[list[dict], dict]:
    """Sorteia pares dev/eval antes de qualquer resposta do modelo.

    ``labels`` mapeia a pasta da classe (fake, true) ao rótulo pedido ao modelo.
    ``pairs`` fixa os pares de cada etapa (ex.: os de outro run) em vez de sortear;
    todos precisam ser elegíveis neste corpus.
    """
    folders = {cls: files_by_id(dataset / texts / cls) for cls in labels}
    common = set.intersection(*(set(files) for files in folders.values()))
    missing = {f"{cls}_only": sorted(set(files) - common, key=id_order)
               for cls, files in folders.items()}

    records: dict[str, list[dict]] = {}
    entries: list[tuple[str, str]] = []
    all_hashes: dict[str, set[str]] = defaultdict(set)
    exclusions: dict[str, list[str]] = defaultdict(list)
    for cls, files in folders.items():
        for pair_id in sorted(set(files) - common, key=id_order):
            path = files[pair_id]
            entries.append((path.relative_to(dataset).as_posix(), sha256(read_text(path))))
    for pair_id in sorted(common, key=id_order):
        pair_records = []
        for cls, label in labels.items():
            path = folders[cls][pair_id]
            content = read_text(path)
            if not content:
                exclusions[pair_id].append(f"vazio:{label}")
            digest = sha256(content)
            all_hashes[digest].add(pair_id)
            relative = path.relative_to(dataset).as_posix()
            entries.append((relative, digest))
            pair_records.append({
                "pair_id": pair_id, "gold_label": label, "relative_path": relative,
                "text_sha256": digest, "characters": len(content),
                "words": len(content.split()),
            })
        if len({row["text_sha256"] for row in pair_records}) < len(pair_records):
            # Textos iguais no mesmo par gerariam a mesma requisição para rótulos distintos.
            exclusions[pair_id].append("texto_identico_no_par")
        records[pair_id] = pair_records

    for repeated_ids in all_hashes.values():
        if len(repeated_ids) > 1:
            for pair_id in repeated_ids:
                exclusions[pair_id].append("texto_identico_em_pares_distintos")

    corpus_fingerprint = fingerprint(entries)
    if expected_fingerprint and corpus_fingerprint != expected_fingerprint:
        raise ValueError(
            f"O corpus em {dataset} difere da versão esperada (fingerprint "
            f"{corpus_fingerprint[:12]}… ≠ {expected_fingerprint[:12]}…). "
            "Rode 'pixi run fetch-data' ou confira a revisão.")

    eligible = sorted(set(records) - set(exclusions), key=id_order)
    if len(eligible) < dev_pairs + eval_pairs:
        raise ValueError(f"Só {len(eligible)} pares elegíveis; requer {dev_pairs + eval_pairs}.")
    if pairs is None:
        random.Random(seed).shuffle(eligible)
        selected = {"dev": eligible[:dev_pairs],
                    "eval": eligible[dev_pairs:dev_pairs + eval_pairs]}
    else:
        selected = {stage: list(pairs.get(stage, [])) for stage in ("dev", "eval")}
        chosen = selected["dev"] + selected["eval"]
        if (len(selected["dev"]), len(selected["eval"])) != (dev_pairs, eval_pairs):
            raise ValueError(f"Pares fixados: dev={len(selected['dev'])}, "
                             f"eval={len(selected['eval'])}; o experimento pede "
                             f"dev={dev_pairs}, eval={eval_pairs}.")
        if len(set(chosen)) != len(chosen):
            raise ValueError("Pares fixados repetidos ou presentes em dev e eval.")
        not_eligible = sorted(set(chosen) - set(eligible), key=id_order)
        if not_eligible:
            raise ValueError(f"Pares fixados ausentes ou excluídos em {texts}: {not_eligible}")
    rows = [{"stage": stage, **row}
            for stage, ids in selected.items() for pair_id in ids for row in records[pair_id]]
    info = {
        "created_at": now_utc(),
        "dataset_revision": dataset_revision(dataset),
        "dataset_fingerprint": corpus_fingerprint,
        "texts": texts, "seed": seed, "dev_pairs": dev_pairs, "eval_pairs": eval_pairs,
        "eligible_pairs": len(eligible),
        "raw_file_counts": {cls: len(files) for cls, files in folders.items()},
        "missing_pair_ids": missing,
        "excluded_pairs": dict(exclusions),
        "population_word_lengths": {
            label: length_summary([int(row["words"]) for pair_id in eligible
                                   for row in records[pair_id] if row["gold_label"] == label])
            for label in labels.values()
        },
        "note": ("Sorteio de pares antes de qualquer resposta do modelo; textos sem truncamento "
                 "adicional." if pairs is None else
                 "Pares fixados (de outro run), não sorteados; textos sem truncamento adicional."),
    }
    return rows, info


def fetch(repository: str, revision: str, destination: Path) -> bool:
    """Baixa o arquivo zip do GitHub numa revisão fixa (sem depender de git).

    Não faz nada se o destino já tiver conteúdo; devolve se baixou.
    """
    if destination.exists() and any(destination.iterdir()):
        return False
    url = f"{repository.rstrip('/').removesuffix('.git')}/archive/{revision}.zip"
    print(f"Baixando {url}", flush=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Extrai ao lado do destino, sem a pasta "<repo>-<revisão>/" do zip: caminhos
    # curtos evitam o limite de 260 caracteres do Windows.
    partial = destination.with_name(f".{destination.name}.partial")
    shutil.rmtree(partial, ignore_errors=True)
    with tempfile.TemporaryFile() as archive:
        with urllib.request.urlopen(url, timeout=120) as response:
            shutil.copyfileobj(response, archive)
        archive.seek(0)
        with zipfile.ZipFile(archive) as bundle:
            for info in bundle.infolist():
                parts = PurePosixPath(info.filename).parts
                if info.filename.startswith("/") or ".." in parts or ":" in info.filename:
                    raise ValueError(f"Caminho inseguro no zip: {info.filename}")
                if len(parts) < 2:
                    continue
                target = partial.joinpath(*parts[1:])
                if info.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(info) as source, target.open("wb") as handle:
                    shutil.copyfileobj(source, handle)
    (partial / SOURCE_FILE).write_text(json.dumps(
        {"repository": repository, "revision": revision, "fetched_at": now_utc()},
        indent=2) + "\n", encoding="utf-8")
    if destination.exists():
        destination.rmdir()
    partial.rename(destination)
    return True
