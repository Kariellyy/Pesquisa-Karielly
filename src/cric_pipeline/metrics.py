from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)


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
