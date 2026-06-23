from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    roc_curve,
)
from torch import nn

from .config import Config
from .modeling import collect_logits, load_checkpoint_model, softmax


def binary_metrics(y: np.ndarray, pred: np.ndarray, score: np.ndarray, threshold: float) -> dict:
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "limiar": float(threshold),
        "acuracia": float((tp + tn) / max(tp + tn + fp + fn, 1)),
        "acuracia_balanceada": float(balanced_accuracy_score(y, pred)),
        "sensibilidade_anormal": float(tp / max(tp + fn, 1)),
        "especificidade_normal": float(tn / max(tn + fp, 1)),
        "f1_anormal": float(f1_score(y, pred, pos_label=1, zero_division=0)),
        "mcc": float(matthews_corrcoef(y, pred)),
        "kappa_cohen": float(cohen_kappa_score(y, pred)),
        "roc_auc": float(roc_auc_score(y, score)),
        "pr_auc_precisao_media": float(average_precision_score(y, score)),
        "brier_score": float(brier_score_loss(y, score)),
        "verdadeiros_normais": int(tn),
        "falsos_anormais": int(fp),
        "falsos_normais": int(fn),
        "verdadeiros_anormais": int(tp),
    }


def youden_threshold(y: np.ndarray, score: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(y, score)
    return float(np.clip(thresholds[np.argmax(tpr - fpr)], 0.0, 1.0))


def sensitivity_threshold(y: np.ndarray, score: np.ndarray, target: float) -> float:
    fpr, tpr, thresholds = roc_curve(y, score)
    ok = tpr >= target
    if not ok.any():
        return float(np.clip(thresholds[np.argmax(tpr)], 0.0, 1.0))
    idx = np.where(ok)[0]
    return float(np.clip(thresholds[idx[np.argmin(fpr[idx])]], 0.0, 1.0))


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    lg = torch.tensor(logits, dtype=torch.float32)
    yy = torch.tensor(y, dtype=torch.long)
    log_t = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_t], lr=0.05, max_iter=100)
    criterion = nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        loss = criterion(lg / torch.exp(log_t), yy)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(torch.exp(log_t).detach().item())


def ece(y: np.ndarray, score: np.ndarray, bins_count: int = 10) -> float:
    bins = np.linspace(0, 1, bins_count + 1)
    total = len(y)
    error = 0.0
    for i in range(bins_count):
        selected = (score >= bins[i]) & (score <= bins[i + 1]) if i == 0 else (score > bins[i]) & (score <= bins[i + 1])
        if selected.sum() == 0:
            continue
        error += (selected.sum() / total) * abs((y[selected] == 1).mean() - score[selected].mean())
    return float(error)


def bootstrap_grouped(y: np.ndarray, score: np.ndarray, groups: np.ndarray, threshold: float, seed: int, replicates: int) -> pd.DataFrame:
    point = _bootstrap_metrics(y, score, threshold)
    rng = np.random.default_rng(seed)
    unique_groups = np.unique(groups)
    index_map = {group: np.where(groups == group)[0] for group in unique_groups}
    samples = {key: [] for key in point}
    for _ in range(replicates):
        chosen = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        idx = np.concatenate([index_map[group] for group in chosen])
        if len(np.unique(y[idx])) < 2:
            continue
        row = _bootstrap_metrics(y[idx], score[idx], threshold)
        for key, value in row.items():
            samples[key].append(value)
    rows = []
    for key, value in point.items():
        lo, hi = np.percentile(samples[key], [2.5, 97.5])
        rows.append({"metrica": key, "valor": round(float(value), 4), "ic95": f"{value:.3f} [{lo:.3f}-{hi:.3f}]"})
    return pd.DataFrame(rows)


def _bootstrap_metrics(y: np.ndarray, score: np.ndarray, threshold: float) -> dict:
    pred = (score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "roc_auc": roc_auc_score(y, score),
        "pr_auc": average_precision_score(y, score),
        "sensibilidade": tp / max(tp + fn, 1),
        "especificidade": tn / max(tn + fp, 1),
        "acuracia_bal": balanced_accuracy_score(y, pred),
        "f1_anormal": f1_score(y, pred, pos_label=1, zero_division=0),
        "mcc": matthews_corrcoef(y, pred),
        "kappa": cohen_kappa_score(y, pred),
    }


def evaluate_partition(config: Config, metadata: pd.DataFrame, force: bool = False) -> None:
    out = config.metrics_dir / "comparacao_limiares.csv"
    if out.exists() and not force:
        print(f"Metricas existentes; pulando avaliacao: {out}")
        return

    val_df = metadata[metadata["particao"] == "validacao"].reset_index(drop=True)
    test_df = metadata[metadata["particao"] == "teste"].reset_index(drop=True)
    model = load_checkpoint_model(config, config.convnext_checkpoint)
    y_val, logits_val = collect_logits(config, model, val_df, tta=True)
    y_test, logits_test = collect_logits(config, model, test_df, tta=True)
    temperature = fit_temperature(logits_val, y_val)
    score_val = softmax(logits_val / temperature)[:, 1]
    score_test = softmax(logits_test / temperature)[:, 1]
    threshold_youden = youden_threshold(y_val, score_val)
    threshold_sens = sensitivity_threshold(y_val, score_val, config.sensitivity_target)
    rows = []
    for name, threshold in [
        ("limiar_0.50", 0.50),
        ("youden", threshold_youden),
        (f"sens>={config.sensitivity_target:.2f}", threshold_sens),
    ]:
        metrics = binary_metrics(y_test, (score_test >= threshold).astype(int), score_test, threshold)
        rows.append(
            {
                "cenario": name,
                "limiar": round(threshold, 3),
                "sensibilidade": round(metrics["sensibilidade_anormal"], 4),
                "especificidade": round(metrics["especificidade_normal"], 4),
                "acuracia_bal": round(metrics["acuracia_balanceada"], 4),
                "f1_anormal": round(metrics["f1_anormal"], 4),
                "mcc": round(metrics["mcc"], 4),
                "roc_auc": round(metrics["roc_auc"], 4),
                "pr_auc": round(metrics["pr_auc_precisao_media"], 4),
                "brier": round(metrics["brier_score"], 4),
            }
        )
    pd.DataFrame(rows).to_csv(out, index=False)
    (config.metrics_dir / "config_avaliacao.json").write_text(
        json.dumps(
            {
                "tta": True,
                "temperatura": temperature,
                "limiar_youden": threshold_youden,
                "limiar_sensibilidade": threshold_sens,
                "alvo_sensibilidade": config.sensitivity_target,
                "ece_antes": ece(y_test, softmax(logits_test)[:, 1]),
                "ece_depois": ece(y_test, score_test),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    bootstrap_grouped(
        y_test,
        score_test,
        test_df["id_imagem"].to_numpy(),
        threshold_youden,
        config.seed,
        config.bootstrap_replicates,
    ).to_csv(config.metrics_dir / "intervalos_confianca_bootstrap.csv", index=False)
    predictions = test_df.copy()
    predictions["prob_anormal"] = score_test
    predictions["alvo_predito"] = (score_test >= threshold_youden).astype(int)
    predictions.to_csv(config.output_dir / "predicoes_teste_artigo.csv", index=False)
    bethesda = (
        predictions.assign(acerto=lambda df: (df["alvo_predito"] == df["alvo"]).astype(int))
        .groupby("classe_bethesda")
        .agg(suporte=("acerto", "size"), acerto_binario=("acerto", "mean"), p_anormal_media=("prob_anormal", "mean"))
    )
    bethesda["acerto_binario_%"] = (bethesda["acerto_binario"] * 100).round(1)
    bethesda["p_anormal_media"] = bethesda["p_anormal_media"].round(3)
    bethesda[["suporte", "acerto_binario_%", "p_anormal_media"]].to_csv(config.metrics_dir / "desempenho_por_bethesda.csv")
    save_learning_curve(config)
    print(pd.DataFrame(rows))


def save_learning_curve(config: Config) -> None:
    history_path = config.convnext_checkpoint.with_suffix(".history.csv")
    if not history_path.exists():
        return
    history = pd.read_csv(history_path)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history["epoca"], history["perda_treino"], label="treino")
    axes[0].plot(history["epoca"], history["perda_validacao"], label="validacao")
    axes[0].set_title("Perda")
    axes[0].legend()
    axes[1].plot(history["epoca"], history["auc_treino"], label="treino")
    axes[1].plot(history["epoca"], history["auc_validacao"], label="validacao")
    axes[1].set_title("AUC ROC")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(config.metrics_dir / "curva_aprendizado.png", dpi=160)
    plt.close(fig)
