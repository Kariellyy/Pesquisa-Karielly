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
    early_stopping_patience: int = 0
    early_stopping_min_delta: float = 0.0

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
    def cv_dir(self) -> Path:
        return self.output_dir / "cv"

    @property
    def cv_assignments_csv(self) -> Path:
        return self.cv_dir / "fold_assignments.csv"

    @property
    def comparison_dir(self) -> Path:
        return self.output_dir / "comparacao_cv"

    @property
    def metadata_csv(self) -> Path:
        return self.output_dir / "metadata_binary.csv"

    @property
    def splits_csv(self) -> Path:
        return self.output_dir / "metadata_splits.csv"

    def model_dir(self, architecture: str) -> Path:
        return self.output_dir / architecture

    def model_checkpoints_dir(self, architecture: str) -> Path:
        return self.model_dir(architecture) / "checkpoints"

    def model_metrics_dir(self, architecture: str) -> Path:
        return self.model_dir(architecture) / "metrics"

    def cv_checkpoint(self, architecture: str, fold: int) -> Path:
        return self.model_checkpoints_dir(architecture) / f"cv_fold_{fold}_{architecture}.pt"

    def cv_folds_metrics(self, architecture: str) -> Path:
        return self.model_metrics_dir(architecture) / "validacao_cruzada_folds.csv"

    def cv_summary_metrics(self, architecture: str) -> Path:
        return self.model_metrics_dir(architecture) / "validacao_cruzada_resumo.csv"


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
        config.cv_dir,
        config.comparison_dir,
        config.model_checkpoints_dir("convnext_tiny"),
        config.model_metrics_dir("convnext_tiny"),
        config.model_checkpoints_dir("resnet50"),
        config.model_metrics_dir("resnet50"),
    ]:
        directory.mkdir(parents=True, exist_ok=True)
