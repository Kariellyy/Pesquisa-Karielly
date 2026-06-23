from __future__ import annotations

import random
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from torchvision.models import ConvNeXt_Tiny_Weights, ResNet50_Weights, convnext_tiny, resnet50
from tqdm import tqdm

from .config import Config
from .data import LABELS


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class CropDataset(Dataset):
    def __init__(self, table: pd.DataFrame, transform=None):
        self.table = table.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.table)

    def __getitem__(self, index: int):
        row = self.table.iloc[index]
        image = Image.open(row.caminho_recorte).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        target = torch.tensor(int(row.alvo), dtype=torch.long)
        return image, target


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def require_cuda(expected_gpu: str) -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA nao esta disponivel. O pipeline foi impedido de rodar na CPU.")
    device = torch.device("cuda:0")
    name = torch.cuda.get_device_name(device)
    if expected_gpu.lower() not in name.lower():
        raise RuntimeError(f"GPU inesperada: {name}. Esperado conter: {expected_gpu}")
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    print(f"GPU ativa: {name} | VRAM {torch.cuda.get_device_properties(device).total_memory / 1024**3:.2f} GB")
    return device


def transforms_train(config: Config):
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(config.image_size, scale=(0.85, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.08, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def transforms_eval(config: Config):
    return transforms.Compose(
        [
            transforms.Resize((config.image_size, config.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def loader_args(config: Config, device: torch.device) -> dict:
    workers = max(0, int(config.num_workers))
    args = {
        "batch_size": config.batch_size,
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": workers > 0,
    }
    if workers > 0:
        args["prefetch_factor"] = 2
    return args


def train_loader(config: Config, table: pd.DataFrame, device: torch.device) -> DataLoader:
    counts = table["alvo"].value_counts().to_dict()
    weights = table["alvo"].map(lambda target: 1.0 / counts[int(target)]).to_numpy(copy=True)
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    return DataLoader(CropDataset(table, transforms_train(config)), sampler=sampler, **loader_args(config, device))


def eval_loader(config: Config, table: pd.DataFrame, device: torch.device) -> DataLoader:
    return DataLoader(CropDataset(table, transforms_eval(config)), shuffle=False, **loader_args(config, device))


def create_convnext() -> nn.Module:
    model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
    features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(features, len(LABELS))
    return model


def create_resnet50() -> nn.Module:
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, len(LABELS))
    return model


def autocast_context(device: torch.device):
    if device.type == "cuda":
        return torch.amp.autocast("cuda", enabled=True)
    return nullcontext()


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def fit_model(
    config: Config,
    model_fn,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    checkpoint_path: Path,
    epochs: int,
    force: bool = False,
    seed: int | None = None,
) -> pd.DataFrame:
    if checkpoint_path.exists() and not force:
        print(f"Checkpoint existente; pulando treino: {checkpoint_path}")
        history_path = checkpoint_path.with_suffix(".history.csv")
        if history_path.exists():
            return pd.read_csv(history_path)
        return pd.DataFrame()

    seed_everything(config.seed if seed is None else seed)
    device = require_cuda(config.expected_gpu)
    model = model_fn().to(device)
    model = model.to(memory_format=torch.channels_last)
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    train_dl = train_loader(config, train_df, device)
    val_dl = eval_loader(config, val_df, device)

    history = []
    best_score = -np.inf
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    history_path = checkpoint_path.with_suffix(".history.csv")

    for epoch in range(1, epochs + 1):
        torch.cuda.reset_peak_memory_stats(device)
        train_metrics = run_epoch(model, train_dl, criterion, optimizer, scaler, device)
        val_metrics = run_epoch(model, val_dl, criterion, None, scaler, device)
        scheduler.step()
        score = val_metrics["auc"] if not np.isnan(val_metrics["auc"]) else val_metrics["accuracy"]
        row = {
            "epoca": epoch,
            "perda_treino": train_metrics["loss"],
            "acuracia_treino": train_metrics["accuracy"],
            "auc_treino": train_metrics["auc"],
            "perda_validacao": val_metrics["loss"],
            "acuracia_validacao": val_metrics["accuracy"],
            "auc_validacao": val_metrics["auc"],
            "taxa_aprendizado": scheduler.get_last_lr()[0],
        }
        history.append(row)
        pd.DataFrame(history).to_csv(history_path, index=False)
        if score > best_score:
            best_score = score
            torch.save(
                {
                    "estado_modelo": model.state_dict(),
                    "nomes_classes": LABELS,
                    "tamanho_recorte": config.image_size,
                    "epoca": epoch,
                    "metricas_validacao": val_metrics,
                },
                checkpoint_path,
            )
        peak = torch.cuda.max_memory_allocated(device) / 1024**3
        print(
            f"Epoca {epoch:02d}/{epochs} | val_auc={val_metrics['auc']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} | pico_vram={peak:.2f} GB"
        )
    return pd.DataFrame(history)


def run_epoch(model, loader, criterion, optimizer, scaler, device: torch.device) -> dict:
    training = optimizer is not None
    model.train(training)
    losses = 0.0
    targets_all = []
    probs_all = []
    for images, targets in tqdm(loader, leave=False):
        images = images.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
        targets = targets.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training), autocast_context(device):
            logits = model(images)
            loss = criterion(logits, targets)
        if training:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        probs = torch.softmax(logits.detach(), dim=1).cpu().numpy()
        probs_all.append(probs)
        targets_all.append(targets.detach().cpu().numpy())
        losses += float(loss.detach().cpu()) * len(targets)
    y = np.concatenate(targets_all)
    p = np.concatenate(probs_all)
    pred = p.argmax(axis=1)
    try:
        auc = roc_auc_score(y, p[:, 1])
    except ValueError:
        auc = np.nan
    return {"loss": losses / len(loader.dataset), "accuracy": float((pred == y).mean()), "auc": float(auc)}


def load_checkpoint_model(config: Config, checkpoint: Path, model_fn=create_convnext) -> nn.Module:
    device = require_cuda(config.expected_gpu)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model = model_fn().to(device)
    model = model.to(memory_format=torch.channels_last)
    model.load_state_dict(payload.get("estado_modelo", payload.get("model_state")))
    model.eval()
    return model


@torch.no_grad()
def collect_logits(config: Config, model: nn.Module, table: pd.DataFrame, tta: bool = False) -> tuple[np.ndarray, np.ndarray]:
    device = require_cuda(config.expected_gpu)
    dl = eval_loader(config, table, device)
    ys = []
    logits_all = []
    model.eval()
    for images, targets in tqdm(dl, desc="Predicao", leave=False):
        images = images.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
        with autocast_context(device):
            logits = model(images).float()
            if tta:
                logits_flip = model(torch.flip(images, dims=[3])).float()
                logits = (logits + logits_flip) / 2.0
        logits_all.append(logits.cpu().numpy())
        ys.append(targets.numpy())
    return np.concatenate(ys), np.concatenate(logits_all)


class ConfigLike:
    def __init__(self, base: Config, **overrides):
        self.__dict__.update(base.__dict__)
        self.__dict__.update(overrides)


def benchmark_batches(config: Config, table: pd.DataFrame, batches: list[int], steps: int = 20) -> list[dict]:
    rows = []
    device = require_cuda(config.expected_gpu)
    for batch in batches:
        bench_config = ConfigLike(config, batch_size=batch, num_workers=0)
        seed_everything(config.seed)
        model = convnext_tiny(weights=None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(LABELS))
        model = model.to(device).to(memory_format=torch.channels_last)
        criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        scaler = torch.amp.GradScaler("cuda", enabled=True)
        loader = train_loader(bench_config, table, device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        start = time.time()
        ok = True
        error = ""
        seen = 0
        try:
            model.train()
            for seen, (images, targets) in enumerate(loader, start=1):
                images = images.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
                targets = targets.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with autocast_context(device):
                    loss = criterion(model(images), targets)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                if seen >= steps:
                    break
            torch.cuda.synchronize(device)
        except RuntimeError as exc:
            ok = False
            error = str(exc).splitlines()[0]
            torch.cuda.empty_cache()
        elapsed = time.time() - start
        peak = torch.cuda.max_memory_allocated(device) / 1024**3
        rows.append(
            {
                "batch_size": batch,
                "ok": ok,
                "steps": seen,
                "seconds": round(elapsed, 2),
                "steps_per_second": round(seen / elapsed, 3) if elapsed > 0 else 0,
                "peak_vram_gb": round(peak, 2),
                "error": error,
            }
        )
        del model, optimizer, scaler, loader
        torch.cuda.empty_cache()
    return rows
