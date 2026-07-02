from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import Config
from .data import load_prepared_metadata


PT_NAMES = {
    "roc_auc": "AUC ROC",
    "pr_auc_precisao_media": "AUC PR",
    "acuracia_balanceada": "Acc. bal.",
    "sensibilidade_anormal": "Sensibilidade",
    "especificidade_normal": "Especificidade",
    "f1_anormal": "F1",
    "mcc": "MCC",
    "kappa_cohen": "Kappa",
    "brier_score": "Brier",
}

MODEL_LABELS = {
    "convnext_tiny": "ConvNeXt-Tiny",
    "resnet50": "ResNet-50",
}

CV_METRICS = [
    "roc_auc",
    "pr_auc_precisao_media",
    "acuracia_balanceada",
    "sensibilidade_anormal",
    "especificidade_normal",
    "f1_anormal",
    "mcc",
    "brier_score",
]


def fmt4(value) -> str:
    if pd.isna(value):
        return "--"
    return f"{float(value):.4f}".replace(".", ",")


def write_table(table: pd.DataFrame, path_base: Path, index: bool = False) -> None:
    path_base.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path_base.with_suffix(".csv"), index=index, encoding="utf-8")
    path_base.with_suffix(".md").write_text(to_markdown(table, index=index), encoding="utf-8")
    path_base.with_suffix(".tex").write_text(table.to_latex(index=index, escape=False), encoding="utf-8")


def to_markdown(table: pd.DataFrame, index: bool = False) -> str:
    data = table.copy()
    if index:
        data = data.reset_index()
    headers = [str(col) for col in data.columns]
    rows = [[str(value) for value in row] for row in data.to_numpy()]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def make_materials(config: Config, output_dir: str | Path = "materiais_artigo") -> None:
    out = Path(output_dir)
    figures = out / "figuras"
    tables = out / "tabelas"
    metrics = out / "metricas"
    data_dir = out / "dados_origem"
    for directory in [figures, tables, metrics, data_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    metadata = load_prepared_metadata(config)
    copy_source_data(config, data_dir)
    copy_reference_figures(figures)
    build_dataset_tables(metadata, tables)
    build_result_tables(config, tables, metrics)
    build_figures(config, metadata, figures)
    write_readme(out)
    print(f"Materiais gerados em: {out.resolve()}")


def copy_source_data(config: Config, data_dir: Path) -> None:
    source_files = [
        config.metadata_csv,
        config.cv_assignments_csv,
        config.comparison_dir / "comparacao_cv_folds.csv",
        config.comparison_dir / "comparacao_cv_resumo.csv",
        config.comparison_dir / "comparacao_cv_pareada.csv",
    ]
    for architecture in MODEL_LABELS:
        source_files.extend([config.cv_folds_metrics(architecture), config.cv_summary_metrics(architecture)])
    for source in source_files:
        if source.exists():
            target_name = f"{source.parent.parent.name}_{source.name}" if source.parent.name == "metrics" else source.name
            shutil.copy2(source, data_dir / target_name)


def copy_reference_figures(figures_dir: Path) -> None:
    article_cells = Path("artigo/figuras/exemplos-celulas-normal-anormal.png")
    if article_cells.exists():
        shutil.copy2(article_cells, figures_dir / "fig_exemplos_celulas_normal_anormal.png")


def build_dataset_tables(metadata: pd.DataFrame, tables: Path) -> None:
    composition = (
        metadata.groupby(["rotulo_binario", "classe_bethesda"])
        .size()
        .reset_index(name="Celulas")
        .rename(columns={"rotulo_binario": "Classe binaria", "classe_bethesda": "Categoria Bethesda"})
    )
    composition["Proporcao (%)"] = (100 * composition["Celulas"] / composition["Celulas"].sum()).round(1)
    write_table(composition, tables / "tabela_composicao_cric")


def build_result_tables(config: Config, tables: Path, metrics_dir: Path) -> None:
    for architecture, model_name in MODEL_LABELS.items():
        cv_summary_path = config.cv_summary_metrics(architecture)
        if cv_summary_path.exists():
            cv_summary = pd.read_csv(cv_summary_path, index_col=0).reset_index(names="Metrica")
            cv_summary["Metrica"] = cv_summary["Metrica"].replace(PT_NAMES)
            cv_summary["Modelo"] = model_name
            cv_summary["Media"] = cv_summary["media"].map(fmt4)
            cv_summary["Desvio"] = cv_summary["desvio_padrao"].map(fmt4)
            cv_summary["Media +/- DP"] = cv_summary["media_pm_dp"].str.replace(".", ",", regex=False)
            write_table(
                cv_summary[["Modelo", "Metrica", "Media", "Desvio", "Media +/- DP"]],
                tables / f"tabela_cv_resumo_{architecture}",
            )

        cv_folds_path = config.cv_folds_metrics(architecture)
        if cv_folds_path.exists():
            cv = pd.read_csv(cv_folds_path)
            cv_table = cv[["fold", *CV_METRICS]].copy()
            cv_table.insert(0, "Modelo", model_name)
            cv_table = cv_table.rename(columns={"fold": "Fold", **PT_NAMES})
            for col in cv_table.columns:
                if col not in {"Modelo", "Fold"}:
                    cv_table[col] = cv_table[col].map(fmt4)
            write_table(cv_table, tables / f"tabela_cv_folds_{architecture}")

    comparison_summary = config.comparison_dir / "comparacao_cv_resumo.csv"
    if comparison_summary.exists():
        comp = pd.read_csv(comparison_summary)
        comp["Metrica"] = comp["metrica"].replace(PT_NAMES)
        comp["Media +/- DP"] = comp["media_pm_dp"].str.replace(".", ",", regex=False)
        table = comp.pivot(index="Metrica", columns="modelo", values="Media +/- DP").reset_index()
        write_table(table, tables / "tabela_comparacao_cv_resumo")

    paired_path = config.comparison_dir / "comparacao_cv_pareada.csv"
    if paired_path.exists():
        paired = pd.read_csv(paired_path)
        cols = ["fold"]
        for metric in ["roc_auc", "acuracia_balanceada", "sensibilidade_anormal", "especificidade_normal", "mcc"]:
            cols.extend(
                [
                    f"{metric}_convnext_tiny",
                    f"{metric}_resnet50",
                    f"delta_resnet50_menos_convnext_tiny_{metric}",
                ]
            )
        table = paired[cols].rename(columns={"fold": "Fold"})
        table = table.rename(columns={col: col.replace("_", " ") for col in table.columns if col != "Fold"})
        for col in table.columns:
            if col != "Fold":
                table[col] = table[col].map(fmt4)
        write_table(table, tables / "tabela_comparacao_cv_pareada")

    build_hyperparameter_table(config, tables)
    write_summary_metrics(config, metrics_dir)


def build_hyperparameter_table(config: Config, tables: Path) -> None:
    table = pd.DataFrame(
        [
            ("Arquitetura principal", "ConvNeXt-Tiny"),
            ("Arquitetura comparada", "ResNet-50"),
            ("Dimensao de entrada", f"{config.image_size}x{config.image_size}x3"),
            ("Batch size", config.batch_size),
            ("Otimizador", "AdamW"),
            ("Taxa de aprendizado", config.learning_rate),
            ("Decaimento de peso", config.weight_decay),
            ("Epocas por fold", config.cv_epochs),
            ("Early stopping", f"paciencia {config.early_stopping_patience}, delta {config.early_stopping_min_delta}"),
            ("Label smoothing", config.label_smoothing),
            ("Folds da CV", config.cv_folds),
            ("Agrupamento dos folds", "id_imagem"),
            ("Limiar por fold", "0,50"),
        ],
        columns=["Parametro", "Valor"],
    )
    write_table(table, tables / "tabela_hiperparametros_pipeline")


def write_summary_metrics(config: Config, metrics_dir: Path) -> None:
    summary = {}
    for name, path in {
        "cv_convnext_tiny": config.cv_summary_metrics("convnext_tiny"),
        "cv_resnet50": config.cv_summary_metrics("resnet50"),
        "comparacao_cv_resumo": config.comparison_dir / "comparacao_cv_resumo.csv",
        "comparacao_cv_pareada": config.comparison_dir / "comparacao_cv_pareada.csv",
    }.items():
        if path.exists():
            summary[name] = pd.read_csv(path).to_dict(orient="records")
    (metrics_dir / "resumo_metricas.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = ["# Resumo das metricas principais", ""]
    for architecture, model_name in MODEL_LABELS.items():
        cv = config.cv_summary_metrics(architecture)
        if cv.exists():
            cv_df = pd.read_csv(cv, index_col=0)
            for metric in ["roc_auc", "acuracia_balanceada", "sensibilidade_anormal", "especificidade_normal"]:
                if metric in cv_df.index:
                    lines.append(f"- {model_name} CV {PT_NAMES.get(metric, metric)}: {cv_df.loc[metric, 'media_pm_dp']}")
    (metrics_dir / "resumo_metricas.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_figures(config: Config, metadata: pd.DataFrame, figures: Path) -> None:
    plt.style.use("default")
    build_composition_figure(metadata, figures)
    build_cv_figure(config, figures)
    build_cv_comparison_figure(config, figures)


def build_composition_figure(metadata: pd.DataFrame, figures: Path) -> None:
    counts = metadata["classe_bethesda"].replace({"Negative for intraepithelial lesion": "NILM"}).value_counts()
    order = ["NILM", "ASC-US", "LSIL", "ASC-H", "HSIL", "SCC"]
    counts = counts.reindex(order).dropna()
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    colors = ["#4C78A8", "#F58518", "#E45756", "#72B7B2", "#54A24B", "#B279A2"]
    ax.bar(counts.index, counts.values, color=colors[: len(counts)])
    ax.set_ylabel("Celulas")
    ax.set_title("Composicao da CRIC por categoria Bethesda")
    for i, value in enumerate(counts.values):
        ax.text(i, value + counts.max() * 0.015, f"{int(value)}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(figures / "fig_composicao_bethesda.png", dpi=220)
    plt.close(fig)


def build_cv_figure(config: Config, figures: Path) -> None:
    comparison = config.comparison_dir / "comparacao_cv_folds.csv"
    if comparison.exists():
        df = pd.read_csv(comparison)
    elif config.cv_folds_metrics("convnext_tiny").exists():
        df = pd.read_csv(config.cv_folds_metrics("convnext_tiny"))
        df["modelo"] = MODEL_LABELS["convnext_tiny"]
    else:
        return

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    colors = {"ConvNeXt-Tiny": "#4C78A8", "ResNet-50": "#F58518"}
    for model, group in df.groupby("modelo", sort=False):
        ax.plot(group["fold"], group["roc_auc"], marker="o", label=model, color=colors.get(model))
    ax.set_xticks(sorted(df["fold"].unique()))
    ax.set_ylim(0.78, 1.0)
    ax.set_xlabel("Fold")
    ax.set_ylabel("AUC ROC")
    ax.set_title("AUC ROC por fold da validacao cruzada")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "fig_validacao_cruzada_folds.png", dpi=220)
    plt.close(fig)


def build_cv_comparison_figure(config: Config, figures: Path) -> None:
    path = config.comparison_dir / "comparacao_cv_folds.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    metrics = ["roc_auc", "pr_auc_precisao_media", "acuracia_balanceada", "sensibilidade_anormal", "especificidade_normal", "mcc"]
    labels = [PT_NAMES[m] for m in metrics]
    x = np.arange(len(metrics))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9.4, 4.8))
    colors = {"ConvNeXt-Tiny": "#4C78A8", "ResNet-50": "#F58518"}
    offsets = {"ConvNeXt-Tiny": -width / 2, "ResNet-50": width / 2}
    for model, group in df.groupby("modelo", sort=False):
        ax.bar(
            x + offsets.get(model, 0),
            group[metrics].mean(),
            width,
            yerr=group[metrics].std(),
            capsize=3,
            label=model,
            color=colors.get(model),
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(0.78, 1.0)
    ax.set_ylabel("Media nos folds")
    ax.set_title("Comparacao por validacao cruzada agrupada")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures / "fig_comparacao_cv_modelos.png", dpi=220)
    plt.close(fig)


def write_readme(out: Path) -> None:
    lines = [
        "# Materiais para o artigo",
        "",
        "Pasta gerada automaticamente a partir dos artefatos de validacao cruzada em `outputs_binary`.",
        "",
        "## Figuras geradas",
        "",
    ]
    for figure in sorted([p.name for p in (out / "figuras").glob("*.png")]):
        lines.append(f"- `figuras/{figure}`")
    lines += ["", "## Tabelas", ""]
    for table in sorted((out / "tabelas").glob("*.csv")):
        lines.append(f"- `tabelas/{table.name}` (+ versoes `.md` e `.tex`)")
    lines += [
        "",
        "## Metricas",
        "",
        "- `metricas/resumo_metricas.md`",
        "- `metricas/resumo_metricas.json`",
        "",
        "## Observacao",
        "",
        "Este material nao inclui o comparativo antigo por treino/validacao/teste fixos. A comparacao do artigo deve usar os 5 folds pareados.",
    ]
    (out / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
