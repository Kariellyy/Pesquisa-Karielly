from __future__ import annotations

import json
import math
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
    "pr_auc": "AUC PR",
    "acuracia_balanceada": "Acc. bal.",
    "sensibilidade_anormal": "Sensibilidade",
    "sensibilidade": "Sensibilidade",
    "especificidade_normal": "Especificidade",
    "especificidade": "Especificidade",
    "f1_anormal": "F1",
    "mcc": "MCC",
    "kappa_cohen": "Kappa",
    "kappa": "Kappa",
    "brier_score": "Brier",
    "brier": "Brier",
}


def fmt4(value) -> str:
    if pd.isna(value):
        return "--"
    return f"{float(value):.4f}".replace(".", ",")


def fmt_pct(value) -> str:
    if pd.isna(value):
        return "--"
    return f"{100 * float(value):.1f}%".replace(".", ",")


def write_table(table: pd.DataFrame, path_base: Path, index: bool = False) -> None:
    path_base.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path_base.with_suffix(".csv"), index=index, encoding="utf-8")
    path_base.with_suffix(".md").write_text(to_markdown(table, index=index), encoding="utf-8")
    path_base.with_suffix(".tex").write_text(
        table.to_latex(index=index, escape=False).replace("\\toprule", "\\toprule"),
        encoding="utf-8",
    )


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


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return (float("nan"), float("nan"))
    phat = successes / total
    denom = 1 + z**2 / total
    center = (phat + z**2 / (2 * total)) / denom
    half = z * math.sqrt((phat * (1 - phat) + z**2 / (4 * total)) / total) / denom
    return center - half, center + half


def copy_existing_figures(config: Config, figures_dir: Path) -> list[str]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    copies = {
        "amostras_predicoes.png": "fig_amostras_predicoes.png",
        "curva_aprendizado.png": "fig_curva_aprendizado.png",
        "curvas_roc_pr_teste.png": "fig_curvas_roc_pr_teste.png",
        "matriz_confusao_teste.png": "fig_matriz_confusao_teste.png",
        "calibracao_confiabilidade.png": "fig_calibracao_confiabilidade.png",
        "desempenho_por_bethesda.png": "fig_bethesda_original.png",
    }
    copied = []
    for source_name, target_name in copies.items():
        source = config.metrics_dir / source_name
        if source.exists():
            shutil.copy2(source, figures_dir / target_name)
            copied.append(target_name)
    article_cells = Path("artigo/figuras/exemplos-celulas-normal-anormal.png")
    if article_cells.exists():
        shutil.copy2(article_cells, figures_dir / "fig_exemplos_celulas_normal_anormal.png")
        copied.append("fig_exemplos_celulas_normal_anormal.png")
    return copied


def make_materials(config: Config, output_dir: str | Path = "materiais_artigo") -> None:
    out = Path(output_dir)
    figures = out / "figuras"
    tables = out / "tabelas"
    metrics = out / "metricas"
    data_dir = out / "dados_origem"
    for directory in [figures, tables, metrics, data_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    metadata = load_prepared_metadata(config)
    copied_figures = copy_existing_figures(config, figures)

    source_files = [
        config.metrics_dir / "comparacao_baseline.csv",
        config.metrics_dir / "comparacao_limiares.csv",
        config.metrics_dir / "validacao_cruzada_folds.csv",
        config.metrics_dir / "validacao_cruzada_resumo.csv",
        config.metrics_dir / "intervalos_confianca_bootstrap.csv",
        config.metrics_dir / "desempenho_por_bethesda.csv",
        config.metrics_dir / "config_avaliacao.json",
        config.output_dir / "predicoes_teste_artigo.csv",
        config.output_dir / "metadata_splits.csv",
    ]
    for source in source_files:
        if source.exists():
            shutil.copy2(source, data_dir / source.name)

    build_dataset_tables(metadata, tables)
    build_result_tables(config, metadata, tables, metrics)
    build_figures(config, metadata, figures)
    write_readme(config, out, copied_figures)
    print(f"Materiais gerados em: {out.resolve()}")


def build_dataset_tables(metadata: pd.DataFrame, tables: Path) -> None:
    composition = (
        metadata.groupby(["rotulo_binario", "classe_bethesda"])
        .size()
        .reset_index(name="Células")
        .rename(columns={"rotulo_binario": "Classe binária", "classe_bethesda": "Categoria Bethesda"})
    )
    composition["Proporção (%)"] = (100 * composition["Células"] / composition["Células"].sum()).round(1)
    write_table(composition, tables / "tabela_composicao_cric")

    splits = (
        metadata.pivot_table(index="particao", columns="rotulo_binario", values="id_celula", aggfunc="count")
        .fillna(0)
        .astype(int)
        .reset_index()
        .rename(columns={"particao": "Partição", "anormal": "Anormal", "normal": "Normal"})
    )
    image_counts = metadata.groupby("particao")["id_imagem"].nunique()
    splits["Imagens"] = splits["Partição"].map(image_counts)
    splits["Total"] = splits[["Anormal", "Normal"]].sum(axis=1)
    splits = splits[["Partição", "Imagens", "Normal", "Anormal", "Total"]]
    write_table(splits, tables / "tabela_particoes")


def build_result_tables(config: Config, metadata: pd.DataFrame, tables: Path, metrics_dir: Path) -> None:
    cv_summary_path = config.metrics_dir / "validacao_cruzada_resumo.csv"
    if cv_summary_path.exists():
        cv_summary = pd.read_csv(cv_summary_path, index_col=0).reset_index(names="Métrica")
        cv_summary["Métrica"] = cv_summary["Métrica"].replace(PT_NAMES)
        cv_summary["Média"] = cv_summary["media"].map(fmt4)
        cv_summary["Desvio"] = cv_summary["desvio_padrao"].map(fmt4)
        cv_summary["Média ± DP"] = cv_summary["media_pm_dp"].str.replace(".", ",", regex=False)
        write_table(cv_summary[["Métrica", "Média", "Desvio", "Média ± DP"]], tables / "tabela_cv_resumo")

    cv_folds_path = config.metrics_dir / "validacao_cruzada_folds.csv"
    if cv_folds_path.exists():
        cv = pd.read_csv(cv_folds_path)
        cv_table = cv[["fold", "roc_auc", "pr_auc_precisao_media", "acuracia_balanceada", "sensibilidade_anormal", "especificidade_normal", "f1_anormal", "mcc", "brier_score"]].copy()
        cv_table = cv_table.rename(columns={"fold": "Fold", **PT_NAMES})
        for col in cv_table.columns:
            if col != "Fold":
                cv_table[col] = cv_table[col].map(fmt4)
        write_table(cv_table, tables / "tabela_cv_folds")

    baseline_path = config.metrics_dir / "comparacao_baseline.csv"
    if baseline_path.exists():
        baseline = pd.read_csv(baseline_path, index_col=0).reset_index(names="Modelo")
        baseline = baseline.rename(columns=PT_NAMES)
        for col in baseline.columns:
            if col != "Modelo":
                baseline[col] = baseline[col].map(fmt4)
        write_table(baseline, tables / "tabela_comparacao_baseline")

    thresholds_path = config.metrics_dir / "comparacao_limiares.csv"
    if thresholds_path.exists():
        thresholds = pd.read_csv(thresholds_path).rename(
            columns={
                "cenario": "Cenário",
                "limiar": "Limiar",
                "sensibilidade": "Sensibilidade",
                "especificidade": "Especificidade",
                "acuracia_bal": "Acc. bal.",
                "f1_anormal": "F1",
                "mcc": "MCC",
                "roc_auc": "AUC ROC",
                "pr_auc": "AUC PR",
                "brier": "Brier",
            }
        )
        for col in thresholds.columns:
            if col != "Cenário":
                thresholds[col] = thresholds[col].map(fmt4)
        write_table(thresholds, tables / "tabela_limiares")

    ic_path = config.metrics_dir / "intervalos_confianca_bootstrap.csv"
    if ic_path.exists():
        ic = pd.read_csv(ic_path).rename(columns={"metrica": "Métrica", "valor": "Valor", "ic95": "IC95%"})
        ic["Métrica"] = ic["Métrica"].replace(PT_NAMES)
        ic["Valor"] = ic["Valor"].map(fmt4)
        ic["IC95%"] = ic["IC95%"].str.replace(".", ",", regex=False)
        write_table(ic, tables / "tabela_ic_bootstrap")

    build_bethesda_table(config, tables)
    build_hyperparameter_table(config, tables)
    write_summary_metrics(config, metrics_dir)


def build_bethesda_table(config: Config, tables: Path) -> None:
    pred_path = config.output_dir / "predicoes_teste_artigo.csv"
    if not pred_path.exists():
        return
    pred = pd.read_csv(pred_path)
    pred["acerto"] = (pred["alvo_predito"] == pred["alvo"]).astype(int)
    rows = []
    order = ["Negative for intraepithelial lesion", "ASC-US", "LSIL", "ASC-H", "HSIL", "SCC"]
    for label in order:
        subset = pred[pred["classe_bethesda"] == label]
        if subset.empty:
            continue
        total = len(subset)
        successes = int(subset["acerto"].sum())
        lo, hi = wilson_interval(successes, total)
        rows.append(
            {
                "Categoria Bethesda": "NILM" if label == "Negative for intraepithelial lesion" else label,
                "n": total,
                "Acerto (%)": round(100 * successes / total, 1),
                "IC95% Wilson": f"{100 * lo:.1f}--{100 * hi:.1f}%".replace(".", ","),
                "P(anormal) média": round(float(subset["prob_anormal"].mean()), 3),
            }
        )
    table = pd.DataFrame(rows)
    write_table(table, tables / "tabela_bethesda_wilson")


def build_hyperparameter_table(config: Config, tables: Path) -> None:
    table = pd.DataFrame(
        [
            ("Arquitetura principal", "ConvNeXt-Tiny"),
            ("Baseline", "ResNet50"),
            ("Dimensão de entrada", f"{config.image_size}x{config.image_size}x3"),
            ("Batch size", config.batch_size),
            ("Otimizador", "AdamW"),
            ("Taxa de aprendizado", config.learning_rate),
            ("Decaimento de peso", config.weight_decay),
            ("Épocas", config.epochs),
            ("Label smoothing", config.label_smoothing),
            ("TTA na partição retida", "Espelho horizontal"),
            ("Folds da CV", config.cv_folds),
        ],
        columns=["Parâmetro", "Valor"],
    )
    write_table(table, tables / "tabela_hiperparametros_pipeline")


def write_summary_metrics(config: Config, metrics_dir: Path) -> None:
    summary = {}
    for name, path in {
        "cv_resumo": config.metrics_dir / "validacao_cruzada_resumo.csv",
        "baseline": config.metrics_dir / "comparacao_baseline.csv",
        "limiares": config.metrics_dir / "comparacao_limiares.csv",
        "bethesda": config.metrics_dir / "desempenho_por_bethesda.csv",
        "avaliacao": config.metrics_dir / "config_avaliacao.json",
    }.items():
        if path.exists():
            if path.suffix == ".json":
                summary[name] = json.loads(path.read_text(encoding="utf-8"))
            else:
                summary[name] = pd.read_csv(path).to_dict(orient="records")
    (metrics_dir / "resumo_metricas.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = ["# Resumo das métricas principais", ""]
    cv = config.metrics_dir / "validacao_cruzada_resumo.csv"
    if cv.exists():
        cv_df = pd.read_csv(cv, index_col=0)
        for metric in ["roc_auc", "pr_auc_precisao_media", "acuracia_balanceada", "sensibilidade_anormal", "especificidade_normal"]:
            if metric in cv_df.index:
                lines.append(f"- CV {PT_NAMES.get(metric, metric)}: {cv_df.loc[metric, 'media_pm_dp']}")
    baseline = config.metrics_dir / "comparacao_baseline.csv"
    if baseline.exists():
        base = pd.read_csv(baseline, index_col=0)
        for model in base.index:
            lines.append(f"- {model} teste 0,50: AUC ROC {base.loc[model, 'roc_auc']:.4f}, Acc. bal. {base.loc[model, 'acuracia_balanceada']:.4f}")
    (metrics_dir / "resumo_metricas.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_figures(config: Config, metadata: pd.DataFrame, figures: Path) -> None:
    plt.style.use("default")
    build_composition_figure(metadata, figures)
    build_baseline_figure(config, figures)
    build_cv_figure(config, figures)
    build_threshold_figure(config, figures)
    build_bethesda_wilson_figure(config, figures)


def build_composition_figure(metadata: pd.DataFrame, figures: Path) -> None:
    counts = metadata["classe_bethesda"].replace({"Negative for intraepithelial lesion": "NILM"}).value_counts()
    order = ["NILM", "ASC-US", "LSIL", "ASC-H", "HSIL", "SCC"]
    counts = counts.reindex(order).dropna()
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    colors = ["#4C78A8", "#F58518", "#E45756", "#72B7B2", "#54A24B", "#B279A2"]
    ax.bar(counts.index, counts.values, color=colors[: len(counts)])
    ax.set_ylabel("Células")
    ax.set_title("Composição da CRIC por categoria Bethesda")
    for i, value in enumerate(counts.values):
        ax.text(i, value + counts.max() * 0.015, f"{int(value)}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(figures / "fig_composicao_bethesda.png", dpi=220)
    plt.close(fig)


def build_baseline_figure(config: Config, figures: Path) -> None:
    path = config.metrics_dir / "comparacao_baseline.csv"
    if not path.exists():
        return
    df = pd.read_csv(path, index_col=0)
    metrics = ["roc_auc", "pr_auc_precisao_media", "acuracia_balanceada", "sensibilidade_anormal", "especificidade_normal", "mcc"]
    labels = [PT_NAMES[m] for m in metrics]
    x = np.arange(len(metrics))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9.4, 4.8))
    ax.bar(x - width / 2, df.loc["ConvNeXt-Tiny", metrics], width, label="ConvNeXt-Tiny", color="#4C78A8")
    ax.bar(x + width / 2, df.loc["ResNet50", metrics], width, label="ResNet50", color="#F58518")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(0.78, 1.0)
    ax.set_ylabel("Valor")
    ax.set_title("Comparação no teste: ConvNeXt-Tiny vs ResNet50")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures / "fig_comparacao_baseline.png", dpi=220)
    plt.close(fig)


def build_cv_figure(config: Config, figures: Path) -> None:
    path = config.metrics_dir / "validacao_cruzada_folds.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    ax.plot(df["fold"], df["roc_auc"], marker="o", label="AUC ROC", color="#4C78A8")
    ax.plot(df["fold"], df["sensibilidade_anormal"], marker="s", label="Sensibilidade", color="#E45756")
    ax.plot(df["fold"], df["especificidade_normal"], marker="^", label="Especificidade", color="#54A24B")
    ax.set_xticks(df["fold"])
    ax.set_ylim(0.78, 1.0)
    ax.set_xlabel("Fold")
    ax.set_ylabel("Valor")
    ax.set_title("Validação cruzada agrupada por imagem")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "fig_validacao_cruzada_folds.png", dpi=220)
    plt.close(fig)


def build_threshold_figure(config: Config, figures: Path) -> None:
    path = config.metrics_dir / "comparacao_limiares.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(8.2, 4.7))
    x = np.arange(len(df))
    width = 0.32
    ax.bar(x - width / 2, df["sensibilidade"], width, label="Sensibilidade", color="#E45756")
    ax.bar(x + width / 2, df["especificidade"], width, label="Especificidade", color="#4C78A8")
    ax.set_xticks(x)
    ax.set_xticklabels(df["cenario"], rotation=10, ha="right")
    ax.set_ylim(0.82, 1.0)
    ax.set_ylabel("Valor")
    ax.set_title("Troca sensibilidade-especificidade por limiar")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures / "fig_limiares_sensibilidade_especificidade.png", dpi=220)
    plt.close(fig)


def build_bethesda_wilson_figure(config: Config, figures: Path) -> None:
    table_path = Path("materiais_artigo/tabelas/tabela_bethesda_wilson.csv")
    if not table_path.exists():
        return
    df = pd.read_csv(table_path)
    y = df["Acerto (%)"].to_numpy()
    lows = []
    highs = []
    for _, row in df.iterrows():
        interval = str(row["IC95% Wilson"]).replace("%", "").replace(",", ".")
        lo, hi = interval.split("--")
        lows.append(float(lo))
        highs.append(float(hi))
    err = np.vstack([y - np.array(lows), np.array(highs) - y])
    fig, ax = plt.subplots(figsize=(8.4, 4.7))
    ax.bar(df["Categoria Bethesda"], y, color="#72B7B2")
    ax.errorbar(df["Categoria Bethesda"], y, yerr=err, fmt="none", ecolor="#222222", capsize=4, linewidth=1)
    ax.set_ylim(65, 103)
    ax.set_ylabel("Acerto binário (%)")
    ax.set_title("Acerto por categoria Bethesda com IC95% de Wilson")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures / "fig_bethesda_wilson.png", dpi=220)
    plt.close(fig)


def write_readme(config: Config, out: Path, copied_figures: list[str]) -> None:
    lines = [
        "# Materiais para o artigo",
        "",
        "Pasta gerada automaticamente a partir de `outputs_binary`.",
        "",
        "## Figuras geradas",
        "",
    ]
    for figure in sorted([p.name for p in (out / "figuras").glob("*.png")]):
        lines.append(f"- `figuras/{figure}`")
    lines += [
        "",
        "## Tabelas",
        "",
    ]
    for table in sorted((out / "tabelas").glob("*.csv")):
        lines.append(f"- `tabelas/{table.name}` (+ versões `.md` e `.tex`)")
    lines += [
        "",
        "## Métricas",
        "",
        "- `metricas/resumo_metricas.md`",
        "- `metricas/resumo_metricas.json`",
        "",
        "## Observação",
        "",
        "Os valores refletem os artefatos atualmente salvos em `outputs_binary`. Se o pipeline for retreinado com `-Force`, regenere estes materiais.",
    ]
    (out / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
