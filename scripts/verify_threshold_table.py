#!/usr/bin/env python3
"""Verify the retained-test threshold table used in the article."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREDICTIONS = ROOT / "materiais_artigo" / "dados_origem" / "predicoes_teste_artigo.csv"
DEFAULT_CONFIG = ROOT / "materiais_artigo" / "dados_origem" / "config_avaliacao.json"


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def confusion(rows: list[dict[str, str]], threshold: float) -> dict[str, float | int]:
    tn = fp = fn = tp = 0
    for row in rows:
        y_true = int(row["alvo"])
        y_pred = 1 if float(row["prob_anormal"]) >= threshold else 0
        if y_true == 0 and y_pred == 0:
            tn += 1
        elif y_true == 0 and y_pred == 1:
            fp += 1
        elif y_true == 1 and y_pred == 0:
            fn += 1
        elif y_true == 1 and y_pred == 1:
            tp += 1
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    return {
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "sensibilidade": sensitivity,
        "especificidade": specificity,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--csv-out", type=Path)
    args = parser.parse_args()

    rows = load_rows(args.predictions)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    thresholds = [
        ("0,50", 0.500),
        ("Youden", round(float(config["limiar_youden"]), 3)),
        ("Alta sens.", round(float(config["limiar_sensibilidade"]), 3)),
    ]

    normal = sum(1 for row in rows if int(row["alvo"]) == 0)
    abnormal = sum(1 for row in rows if int(row["alvo"]) == 1)
    print(f"predictions={args.predictions}")
    print(f"config={args.config}")
    print(f"test_normal={normal} test_abnormal={abnormal} total={len(rows)}")
    print("Regime,Limiar,TN,FP,FN,TP,Sensibilidade,Especificidade")

    output_rows = []
    for regime, threshold in thresholds:
        result = confusion(rows, threshold)
        if result["tn"] + result["fp"] != normal:
            raise SystemExit(f"normal-count mismatch for {regime}")
        if result["fn"] + result["tp"] != abnormal:
            raise SystemExit(f"abnormal-count mismatch for {regime}")
        output_row = {
            "Regime": regime,
            "Limiar": f"{threshold:.3f}",
            "TN": str(result["tn"]),
            "FP": str(result["fp"]),
            "FN": str(result["fn"]),
            "TP": str(result["tp"]),
            "Sensibilidade": f"{result['sensibilidade']:.4f}",
            "Especificidade": f"{result['especificidade']:.4f}",
        }
        output_rows.append(output_row)
        print(",".join(output_row.values()))

    if args.csv_out:
        args.csv_out.parent.mkdir(parents=True, exist_ok=True)
        with args.csv_out.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
            writer.writeheader()
            writer.writerows(output_rows)


if __name__ == "__main__":
    main()
