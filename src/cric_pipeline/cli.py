from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

from .config import ensure_dirs, load_config
from .data import load_prepared_metadata, prepare_data
from .metrics import binary_metrics, evaluate_partition
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


def cmd_train(args) -> None:
    config = load_config(args.config)
    metadata = load_prepared_metadata(config)
    train_df = metadata[metadata["particao"] == "treino"].reset_index(drop=True)
    val_df = metadata[metadata["particao"] == "validacao"].reset_index(drop=True)
    fit_model(config, create_convnext, train_df, val_df, config.convnext_checkpoint, config.epochs, force=args.force)


def cmd_eval(args) -> None:
    config = load_config(args.config)
    metadata = load_prepared_metadata(config)
    evaluate_partition(config, metadata, force=args.force)


def cmd_cv(args) -> None:
    config = load_config(args.config)
    metadata = load_prepared_metadata(config)
    out_folds = config.metrics_dir / "validacao_cruzada_folds.csv"
    done = set()
    rows = []
    if out_folds.exists() and not args.force:
        current = pd.read_csv(out_folds)
        rows = current.to_dict("records")
        done = set(current["fold"].astype(int).tolist())
        print(f"Retomando CV; folds ja concluidos: {sorted(done)}")
    elif out_folds.exists() and args.force:
        out_folds.unlink()

    sgkf = StratifiedGroupKFold(n_splits=config.cv_folds, shuffle=True, random_state=config.seed)
    start = time.time()
    for fold, (train_idx, holdout_idx) in enumerate(
        sgkf.split(metadata, y=metadata["alvo"], groups=metadata["id_imagem"]), start=1
    ):
        if fold in done:
            continue
        full_train = metadata.iloc[train_idx].reset_index(drop=True)
        holdout = metadata.iloc[holdout_idx].reset_index(drop=True)
        inner_train_idx, inner_val_idx = next(
            GroupShuffleSplit(1, test_size=0.15, random_state=config.seed + fold).split(
                full_train, groups=full_train["id_imagem"]
            )
        )
        fold_ckpt = config.checkpoints_dir / f"cv_fold_{fold}_convnext_tiny.pt"
        print(f"=== Fold {fold}/{config.cv_folds} | holdout={len(holdout)} celulas ===")
        fit_model(
            config,
            create_convnext,
            full_train.iloc[inner_train_idx],
            full_train.iloc[inner_val_idx],
            fold_ckpt,
            config.cv_epochs,
            force=args.force,
            seed=config.seed + fold,
        )
        model = load_checkpoint_model(config, fold_ckpt)
        y, logits = collect_logits(config, model, holdout, tta=False)
        score = softmax(logits)[:, 1]
        metrics = binary_metrics(y, (score >= 0.50).astype(int), score, 0.50)
        metrics["fold"] = fold
        rows.append(metrics)
        pd.DataFrame(rows).to_csv(out_folds, index=False)
        print(
            f"fold {fold}: auc={metrics['roc_auc']:.4f} "
            f"acc_bal={metrics['acuracia_balanceada']:.4f} "
            f"sens={metrics['sensibilidade_anormal']:.4f} "
            f"spec={metrics['especificidade_normal']:.4f}"
        )

    cv = pd.DataFrame(rows).sort_values("fold")
    keys = [
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
    summary = cv[keys].agg(["mean", "std"]).T
    summary.columns = ["media", "desvio_padrao"]
    summary["media_pm_dp"] = summary.apply(lambda row: f"{row['media']:.4f} +/- {row['desvio_padrao']:.4f}", axis=1)
    summary.to_csv(config.metrics_dir / "validacao_cruzada_resumo.csv")
    print(summary)
    print(f"Tempo CV: {(time.time() - start) / 60:.1f} min")


def cmd_baseline(args) -> None:
    config = load_config(args.config)
    metadata = load_prepared_metadata(config)
    train_df = metadata[metadata["particao"] == "treino"].reset_index(drop=True)
    val_df = metadata[metadata["particao"] == "validacao"].reset_index(drop=True)
    test_df = metadata[metadata["particao"] == "teste"].reset_index(drop=True)
    fit_model(config, create_resnet50, train_df, val_df, config.resnet_checkpoint, config.epochs, force=args.force)
    rows = {}
    for name, checkpoint, factory in [
        ("ConvNeXt-Tiny", config.convnext_checkpoint, create_convnext),
        ("ResNet50", config.resnet_checkpoint, create_resnet50),
    ]:
        model = load_checkpoint_model(config, checkpoint, factory)
        y, logits = collect_logits(config, model, test_df, tta=False)
        score = softmax(logits)[:, 1]
        metrics = binary_metrics(y, (score >= 0.50).astype(int), score, 0.50)
        rows[name] = {
            key: round(metrics[key], 4)
            for key in [
                "roc_auc",
                "pr_auc_precisao_media",
                "acuracia_balanceada",
                "sensibilidade_anormal",
                "especificidade_normal",
                "f1_anormal",
                "mcc",
                "brier_score",
            ]
        }
    table = pd.DataFrame(rows).T
    table.to_csv(config.metrics_dir / "comparacao_baseline.csv")
    print(table)


def cmd_all(args) -> None:
    cmd_prepare(args)
    cmd_train(args)
    cmd_eval(args)
    cmd_cv(args)
    cmd_baseline(args)


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
        ("convnext_checkpoint", config.convnext_checkpoint),
        ("resnet_checkpoint", config.resnet_checkpoint),
        ("partition_metrics", config.metrics_dir / "comparacao_limiares.csv"),
        ("cv_folds", config.metrics_dir / "validacao_cruzada_folds.csv"),
        ("cv_summary", config.metrics_dir / "validacao_cruzada_resumo.csv"),
        ("baseline", config.metrics_dir / "comparacao_baseline.csv"),
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
    parser = argparse.ArgumentParser(description="Pipeline local CRIC ConvNeXt/ResNet")
    sub = parser.add_subparsers(dest="command", required=True)
    commands = {
        "check": cmd_check,
        "status": cmd_status,
        "benchmark": cmd_benchmark,
        "prepare": cmd_prepare,
        "train": cmd_train,
        "eval": cmd_eval,
        "cv": cmd_cv,
        "baseline": cmd_baseline,
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
        p.set_defaults(func=fn)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
