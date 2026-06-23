from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    seed: int
    data_dir: Path
    output_dir: Path
    image_size: int
    batch_size: int
    num_workers: int
    epochs: int
    learning_rate: float
    weight_decay: float
    label_smoothing: float
    expected_gpu: str
    sensitivity_target: float
    bootstrap_replicates: int
    cv_folds: int
    cv_epochs: int

    @property
    def images_dir(self) -> Path:
        return self.data_dir / "images"

    @property
    def annotations_csv(self) -> Path:
        return self.data_dir / "classification" / "classifications.csv"

    @property
    def crops_dir(self) -> Path:
        return self.output_dir / "crops"

    @property
    def checkpoints_dir(self) -> Path:
        return self.output_dir / "checkpoints"

    @property
    def metrics_dir(self) -> Path:
        return self.output_dir / "metrics"

    @property
    def metadata_csv(self) -> Path:
        return self.output_dir / "metadata_binary.csv"

    @property
    def splits_csv(self) -> Path:
        return self.output_dir / "metadata_splits.csv"

    @property
    def convnext_checkpoint(self) -> Path:
        return self.checkpoints_dir / "best_convnext_tiny_binary.pt"

    @property
    def resnet_checkpoint(self) -> Path:
        return self.checkpoints_dir / "best_resnet50_binary.pt"


def load_config(path: str | Path) -> Config:
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    root = path.resolve().parents[1]
    raw["data_dir"] = (root / raw["data_dir"]).resolve()
    raw["output_dir"] = (root / raw["output_dir"]).resolve()
    return Config(**raw)


def ensure_dirs(config: Config) -> None:
    for directory in [
        config.output_dir,
        config.crops_dir,
        config.checkpoints_dir,
        config.metrics_dir,
    ]:
        directory.mkdir(parents=True, exist_ok=True)
