from __future__ import annotations

import argparse
import shutil
import time

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

from .config import ensure_dirs, load_config
from .data import load_prepared_metadata, prepare_data
from .metrics import binary_metrics
from .materials import make_materials
from .modeling import (
    benchmark_batches,
    collect_logits,
    create_convnext,
    create_resnet50,
    fit_model,
    load_checkpoint_model,
    require_cuda,
    softmax,
)


ARCHITECTURES = {
    "convnext_tiny": {
        "label": "ConvNeXt-Tiny",
        "factory": create_convnext,
    },
    "resnet50": {
        "label": "ResNet-50",
        "factory": create_resnet50,
    },
}

CV_METRIC_KEYS = [
    "roc_auc",
    "pr_auc_precisao_media",
    "acuracia_balanceada",
    "sensibilidade_anormal",
    "especificidade_normal",
    "f1_anormal",
    "mcc",
    "kappa_cohen",
    "brier_score",
]


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default="configs/local_3060.json")
    parser.add_argument("--force", action="store_true")


def cmd_check(args) -> None:
    config = load_config(args.config)
    require_cuda(config.expected_gpu)
    print("GPU_READY")


def cmd_prepare(args) -> None:
    config = load_config(args.config)
    ensure_dirs(config)
    prepare_data(config, force_crops=args.force, force_splits=args.force)


def get_or_create_cv_assignments(config, metadata: pd.DataFrame, force: bool = False) -> pd.DataFrame:
    if config.cv_assignments_csv.exists() and not force:
        return pd.read_csv(config.cv_assignments_csv)

    rows = []
    sgkf = StratifiedGroupKFold(n_splits=config.cv_folds, shuffle=True, random_state=config.seed)
    for fold, (_, holdout_idx) in enumerate(
        sgkf.split(metadata, y=metadata["alvo"], groups=metadata["id_imagem"]), start=1
    ):
        holdout = metadata.iloc[holdout_idx]
        for image_id, group in holdout.groupby("id_imagem"):
            rows.append(
                {
                    "id_imagem": int(image_id),
                    "fold": int(fold),
                    "celulas": int(len(group)),
                    "normais": int((group["alvo"] == 0).sum()),
                    "anormais": int((group["alvo"] == 1).sum()),
                }
            )
    assignments = pd.DataFrame(rows).sort_values(["fold", "id_imagem"]).reset_index(drop=True)
    config.cv_assignments_csv.parent.mkdir(parents=True, exist_ok=True)
    assignments.to_csv(config.cv_assignments_csv, index=False)
    return assignments


def summarize_cv(rows: list[dict]) -> pd.DataFrame:
    cv = pd.DataFrame(rows).sort_values("fold")
    summary = cv[CV_METRIC_KEYS].agg(["mean", "std"]).T
    summary.columns = ["media", "desvio_padrao"]
    summary["media_pm_dp"] = summary.apply(lambda row: f"{row['media']:.4f} +/- {row['desvio_padrao']:.4f}", axis=1)
    return summary


def run_cv_for_architecture(config, metadata: pd.DataFrame, architecture: str, force: bool = False) -> None:
    spec = ARCHITECTURES[architecture]
    label = spec["label"]
    factory = spec["factory"]
    assignments = get_or_create_cv_assignments(config, metadata, force=False)
    out_folds = config.cv_folds_metrics(architecture)
    out_summary = config.cv_summary_metrics(architecture)
    out_folds.parent.mkdir(parents=True, exist_ok=True)

    done = set()
    rows = []
    if out_folds.exists() and not force:
        current = pd.read_csv(out_folds)
        rows = current.to_dict("records")
        done = set(current["fold"].astype(int).tolist())
        print(f"Retomando CV {label}; folds ja concluidos: {sorted(done)}")
    elif out_folds.exists() and force:
        out_folds.unlink()

    start = time.time()
    for fold in range(1, config.cv_folds + 1):
        if fold in done:
            continue
        holdout_images = set(assignments.loc[assignments["fold"] == fold, "id_imagem"].astype(int))
        holdout = metadata[metadata["id_imagem"].astype(int).isin(holdout_images)].reset_index(drop=True)
        full_train = metadata[~metadata["id_imagem"].astype(int).isin(holdout_images)].reset_index(drop=True)
        inner_train_idx, inner_val_idx = next(
            GroupShuffleSplit(1, test_size=0.15, random_state=config.seed + fold).split(
                full_train, groups=full_train["id_imagem"]
            )
        )
        fold_ckpt = config.cv_checkpoint(architecture, fold)
        print(f"=== {label} | Fold {fold}/{config.cv_folds} | holdout={len(holdout)} celulas ===")
        fit_model(
            config,
            factory,
            full_train.iloc[inner_train_idx],
            full_train.iloc[inner_val_idx],
            fold_ckpt,
            config.cv_epochs,
            force=force,
            seed=config.seed + fold,
        )
        model = load_checkpoint_model(config, fold_ckpt, factory)
        y, logits = collect_logits(config, model, holdout, tta=False)
        score = softmax(logits)[:, 1]
        metrics = binary_metrics(y, (score >= 0.50).astype(int), score, 0.50)
        metrics["fold"] = fold
        metrics["modelo"] = label
        metrics["arquitetura"] = architecture
        rows.append(metrics)
        pd.DataFrame(rows).sort_values("fold").to_csv(out_folds, index=False)
        print(
            f"{label} fold {fold}: auc={metrics['roc_auc']:.4f} "
            f"acc_bal={metrics['acuracia_balanceada']:.4f} "
            f"sens={metrics['sensibilidade_anormal']:.4f} "
            f"spec={metrics['especificidade_normal']:.4f}"
        )

    summary = summarize_cv(rows)
    summary.to_csv(out_summary)
    print(summary)
    print(f"Tempo CV {label}: {(time.time() - start) / 60:.1f} min")


def cmd_cv(args) -> None:
    config = load_config(args.config)
    metadata = load_prepared_metadata(config)
    architecture = getattr(args, "architecture", "convnext_tiny")
    run_cv_for_architecture(config, metadata, architecture, force=args.force)


def cmd_compare_cv(args) -> None:
    config = load_config(args.config)
    missing = [arch for arch in ARCHITECTURES if not config.cv_folds_metrics(arch).exists()]
    if missing:
        names = ", ".join(ARCHITECTURES[arch]["label"] for arch in missing)
        raise FileNotFoundError(f"CV ainda nao encontrada para: {names}")

    frames = []
    for architecture, spec in ARCHITECTURES.items():
        table = pd.read_csv(config.cv_folds_metrics(architecture))
        table["modelo"] = spec["label"]
        table["arquitetura"] = architecture
        frames.append(table)
    combined = pd.concat(frames, ignore_index=True)
    config.comparison_dir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(config.comparison_dir / "comparacao_cv_folds.csv", index=False)

    summary_rows = []
    for (architecture, model), group in combined.groupby(["arquitetura", "modelo"], sort=False):
        for metric in CV_METRIC_KEYS:
            summary_rows.append(
                {
                    "arquitetura": architecture,
                    "modelo": model,
                    "metrica": metric,
                    "media": float(group[metric].mean()),
                    "desvio_padrao": float(group[metric].std()),
                    "media_pm_dp": f"{group[metric].mean():.4f} +/- {group[metric].std():.4f}",
                }
            )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(config.comparison_dir / "comparacao_cv_resumo.csv", index=False)

    conv = combined[combined["arquitetura"] == "convnext_tiny"].set_index("fold")
    res = combined[combined["arquitetura"] == "resnet50"].set_index("fold")
    paired = pd.DataFrame({"fold": sorted(set(conv.index) & set(res.index))})
    for metric in CV_METRIC_KEYS:
        paired[f"{metric}_convnext_tiny"] = paired["fold"].map(conv[metric])
        paired[f"{metric}_resnet50"] = paired["fold"].map(res[metric])
        paired[f"delta_resnet50_menos_convnext_tiny_{metric}"] = (
            paired[f"{metric}_resnet50"] - paired[f"{metric}_convnext_tiny"]
        )
    paired.to_csv(config.comparison_dir / "comparacao_cv_pareada.csv", index=False)
    print(summary.pivot(index="metrica", columns="modelo", values="media_pm_dp"))


def cmd_all(args) -> None:
    cmd_prepare(args)
    config = load_config(args.config)
    metadata = load_prepared_metadata(config)
    run_cv_for_architecture(config, metadata, "convnext_tiny", force=args.force)
    run_cv_for_architecture(config, metadata, "resnet50", force=args.force)
    cmd_compare_cv(args)


def cmd_package(args) -> None:
    config = load_config(args.config)
    target = shutil.make_archive(str(config.output_dir.with_name("outputs_binary_copy")), "zip", root_dir=config.output_dir.parent, base_dir=config.output_dir.name)
    print(f"Pacote salvo em: {target}")


def cmd_materials(args) -> None:
    config = load_config(args.config)
    make_materials(config, output_dir=args.output)


def cmd_status(args) -> None:
    config = load_config(args.config)
    artifacts = [
        ("metadata", config.metadata_csv),
        ("splits", config.splits_csv),
        ("cv_fold_assignments", config.cv_assignments_csv),
        ("convnext_cv_folds", config.cv_folds_metrics("convnext_tiny")),
        ("convnext_cv_summary", config.cv_summary_metrics("convnext_tiny")),
        ("resnet50_cv_folds", config.cv_folds_metrics("resnet50")),
        ("resnet50_cv_summary", config.cv_summary_metrics("resnet50")),
        ("cv_comparison_folds", config.comparison_dir / "comparacao_cv_folds.csv"),
        ("cv_comparison_summary", config.comparison_dir / "comparacao_cv_resumo.csv"),
        ("cv_comparison_paired", config.comparison_dir / "comparacao_cv_pareada.csv"),
    ]
    for name, path in artifacts:
        if path.exists():
            size_mb = path.stat().st_size / 1024**2
            print(f"[ok]      {name:20s} {path} ({size_mb:.2f} MB)")
        else:
            print(f"[faltando] {name:20s} {path}")


def cmd_benchmark(args) -> None:
    config = load_config(args.config)
    metadata = load_prepared_metadata(config)
    train_df = metadata[metadata["particao"] == "treino"].reset_index(drop=True)
    rows = benchmark_batches(config, train_df, batches=args.batches, steps=args.steps)
    table = pd.DataFrame(rows)
    out = config.metrics_dir / "benchmark_batch_gpu.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    print(table)
    print(f"Benchmark salvo em: {out}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pipeline local CRIC com CV agrupada ConvNeXt/ResNet")
    sub = parser.add_subparsers(dest="command", required=True)
    commands = {
        "check": cmd_check,
        "status": cmd_status,
        "benchmark": cmd_benchmark,
        "prepare": cmd_prepare,
        "cv": cmd_cv,
        "cv-convnext": cmd_cv,
        "cv-resnet50": cmd_cv,
        "compare-cv": cmd_compare_cv,
        "all": cmd_all,
        "package": cmd_package,
        "materials": cmd_materials,
    }
    for name, fn in commands.items():
        p = sub.add_parser(name)
        add_common(p)
        if name == "benchmark":
            p.add_argument("--batches", nargs="+", type=int, default=[24, 48, 64])
            p.add_argument("--steps", type=int, default=20)
        if name == "materials":
            p.add_argument("--output", default="materiais_artigo")
        if name == "cv":
            p.set_defaults(architecture="convnext_tiny")
        if name == "cv-convnext":
            p.set_defaults(architecture="convnext_tiny")
        if name == "cv-resnet50":
            p.set_defaults(architecture="resnet50")
        p.set_defaults(func=fn)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
