"""Métricas de classificação com respostas inválidas contadas como erro."""

from __future__ import annotations

from collections import Counter, defaultdict

from .prompts import INVALID


def summarize(rows: list[dict], labels: tuple[str, ...]) -> dict:
    """Métricas de um conjunto de respostas.

    ``INVALIDA`` conta como erro em acurácia, recall e F1; o MCC usa apenas as
    respostas válidas (``mcc_valid_only``).
    """
    from sklearn.metrics import matthews_corrcoef

    predictions = (*labels, INVALID)
    counts = Counter((row["gold_label"], row["prediction"]) for row in rows)
    class_metrics = {}
    for label in labels:
        tp = counts[label, label]
        fp = sum(counts[other, label] for other in labels if other != label)
        support = sum(counts[label, pred] for pred in predictions)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        class_metrics[label] = {
            "support": support, "precision": precision, "recall": recall, "f1": f1,
        }
    valid = [row for row in rows if row["prediction"] in labels]
    return {
        "n": len(rows), "valid": len(valid),
        "valid_rate": len(valid) / len(rows) if rows else 0.0,
        "accuracy": (sum(row["gold_label"] == row["prediction"] for row in rows) / len(rows)
                     if rows else 0.0),
        "balanced_accuracy": sum(class_metrics[label]["recall"] for label in labels) / len(labels),
        "macro_f1": sum(class_metrics[label]["f1"] for label in labels) / len(labels),
        "mcc_valid_only": (
            float(matthews_corrcoef([r["gold_label"] for r in valid],
                                    [r["prediction"] for r in valid]))
            if len(valid) >= 2 else None
        ),
        "per_class": class_metrics,
        "confusion": {
            label: {pred: counts[label, pred] for pred in predictions}
            for label in labels
        },
    }


def fit_length_rule(train: list[tuple[int, str]], labels: tuple[str, str]) -> dict:
    """Melhor regra "palavras > limiar => rótulo" em ``train`` (pares palavras, rótulo).

    Testa os dois sentidos e todos os limiares observados; em empate, o menor limiar.
    """
    import numpy as np

    if len(labels) != 2 or not train:
        raise ValueError("A regra de comprimento requer dois rótulos e dados de ajuste.")
    words = np.array([w for w, _ in train])
    first = np.array([label == labels[0] for _, label in train])
    thresholds = np.unique(words)
    # Quantos de cada rótulo têm palavras <= limiar.
    order = np.argsort(words, kind="stable")
    ends = np.searchsorted(words[order], thresholds, side="right")
    first_le = np.concatenate([[0], np.cumsum(first[order])])[ends]
    second_le = ends - first_le
    n_first, n_second = int(first.sum()), int((~first).sum())
    # accuracy[above][i]: acerto com limiar thresholds[i] e "longo => labels[above]".
    accuracy = {1: (first_le + n_second - second_le) / len(train),
                0: (second_le + n_first - first_le) / len(train)}
    above = max(accuracy, key=lambda side: (accuracy[side].max(), side))
    index = int(np.argmax(accuracy[above]))  # primeira ocorrência = menor limiar
    return {"threshold_words": int(thresholds[index]), "above_label": labels[above],
            "below_label": labels[1 - above], "fit_accuracy": float(accuracy[above][index]),
            "fit_texts": len(train)}


def apply_length_rule(rule: dict, words: int) -> str:
    return rule["above_label"] if words > rule["threshold_words"] else rule["below_label"]


def pair_bootstrap(rows: list[dict], labels: tuple[str, ...], seed: int,
                   repeats: int = 2000) -> list[float]:
    """IC 95% do Macro-F1 reamostrando pares (os dois documentos de cada par juntos)."""
    import numpy as np

    by_pair = defaultdict(list)
    for row in rows:
        by_pair[row["pair_id"]].append(row)
    if any(len(group) != 2 for group in by_pair.values()):
        raise ValueError("Bootstrap requer os dois documentos de cada par.")
    ids = list(by_pair)
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(repeats):
        sample = [row for index in rng.integers(0, len(ids), size=len(ids))
                  for row in by_pair[ids[int(index)]]]
        estimates.append(summarize(sample, labels)["macro_f1"])
    return [float(value) for value in np.percentile(estimates, [2.5, 97.5])]
