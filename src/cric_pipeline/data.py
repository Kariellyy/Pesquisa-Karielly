from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm

from .config import Config, ensure_dirs


BETHESDA_NORMAL = "Negative for intraepithelial lesion"
LABELS = ["normal", "anormal"]


OLD_COLUMNS = {
    "image_id": "id_imagem",
    "image_filename": "arquivo_imagem",
    "cell_id": "id_celula",
    "bethesda_system": "classe_bethesda",
    "binary_label": "rotulo_binario",
    "target": "alvo",
    "nucleus_x": "nucleo_x",
    "nucleus_y": "nucleo_y",
    "image_path": "caminho_imagem",
    "crop_path": "caminho_recorte",
    "split": "particao",
}
OLD_LABELS = {"abnormal": "anormal", "normal": "normal"}
OLD_SPLITS = {"train": "treino", "val": "validacao", "test": "teste"}


def standardize_metadata(table: pd.DataFrame) -> pd.DataFrame:
    table = table.rename(columns=OLD_COLUMNS).copy()
    if "rotulo_binario" in table.columns:
        table["rotulo_binario"] = table["rotulo_binario"].replace(OLD_LABELS)
    if "particao" in table.columns:
        table["particao"] = table["particao"].replace(OLD_SPLITS)
    return table


def normalize_metadata_paths(config: Config, table: pd.DataFrame) -> pd.DataFrame:
    table = standardize_metadata(table)
    if "caminho_recorte" in table.columns:
        table["caminho_recorte"] = [
            str(resolve_crop_path(config, row))
            for row in table.itertuples(index=False)
        ]
    if "caminho_imagem" in table.columns:
        table["caminho_imagem"] = [
            str(resolve_image_path(config, row))
            for row in table.itertuples(index=False)
        ]
    return table


def resolve_crop_path(config: Config, row) -> Path:
    current = Path(str(row.caminho_recorte))
    if current.exists():
        return current
    label = str(row.rotulo_binario)
    candidate = config.crops_dir / label / current.name
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Recorte nao encontrado: {current} nem {candidate}")


def resolve_image_path(config: Config, row) -> Path:
    current = Path(str(row.caminho_imagem))
    if current.exists():
        return current
    candidate = config.images_dir / current.name
    if candidate.exists():
        return candidate
    return find_image_path(config, int(row.id_imagem), str(row.arquivo_imagem))


def find_image_path(config: Config, image_id: int, image_file: str) -> Path:
    suffix = Path(str(image_file)).name
    candidates = sorted(config.images_dir.glob(f"cric_image_{int(image_id):03d}_*.png"))
    for candidate in candidates:
        if candidate.name.endswith(suffix):
            return candidate
    alternatives = sorted(config.images_dir.glob(f"*{Path(suffix).stem}*.png"))
    if alternatives:
        return alternatives[0]
    raise FileNotFoundError(f"Imagem nao encontrada: image_id={image_id}, file={image_file}")


def load_annotations(config: Config) -> pd.DataFrame:
    if not config.annotations_csv.exists():
        raise FileNotFoundError(f"CSV da CRIC nao encontrado: {config.annotations_csv}")
    annotations = pd.read_csv(config.annotations_csv)
    annotations["rotulo_binario"] = np.where(
        annotations["bethesda_system"].eq(BETHESDA_NORMAL), "normal", "anormal"
    )
    annotations["alvo"] = annotations["rotulo_binario"].map({"normal": 0, "anormal": 1}).astype(int)

    image_map: dict[tuple[int, str], Path] = {}
    for row in annotations[["image_id", "image_filename"]].drop_duplicates().itertuples(index=False):
        key = (int(row.image_id), str(row.image_filename))
        image_map[key] = find_image_path(config, row.image_id, row.image_filename)

    annotations["caminho_imagem"] = [
        str(image_map[(int(row.image_id), str(row.image_filename))])
        for row in annotations.itertuples(index=False)
    ]
    return annotations


def crop_centered(image: Image.Image, center_x: int, center_y: int, size: int) -> Image.Image:
    cx = int(center_x) - 1
    cy = int(center_y) - 1
    half = size // 2
    left = cx - half
    top = cy - half
    right = left + size
    bottom = top + size

    src_left = max(left, 0)
    src_top = max(top, 0)
    src_right = min(right, image.width)
    src_bottom = min(bottom, image.height)

    crop = Image.new("RGB", (size, size), (255, 255, 255))
    patch = image.crop((src_left, src_top, src_right, src_bottom))
    crop.paste(patch, (src_left - left, src_top - top))
    return crop


def generate_crops(config: Config, force: bool = False) -> pd.DataFrame:
    ensure_dirs(config)
    if config.metadata_csv.exists() and not force:
        print(f"Usando metadados existentes: {config.metadata_csv}")
        return normalize_metadata_paths(config, pd.read_csv(config.metadata_csv))

    annotations = load_annotations(config)
    rows = []
    groups = annotations.groupby("caminho_imagem", sort=False)
    for image_path, group in tqdm(groups, total=len(groups), desc="Recortando imagens"):
        image = Image.open(image_path).convert("RGB")
        for row in group.itertuples(index=False):
            label = row.rotulo_binario
            crop_dir = config.crops_dir / label
            crop_dir.mkdir(parents=True, exist_ok=True)
            crop_name = f"img{int(row.image_id):03d}_cell{int(row.cell_id):04d}_{label}.png"
            crop_path = crop_dir / crop_name
            if force or not crop_path.exists():
                crop = crop_centered(image, row.nucleus_x, row.nucleus_y, config.image_size)
                crop.save(crop_path)
            rows.append(
                {
                    "id_imagem": int(row.image_id),
                    "arquivo_imagem": row.image_filename,
                    "id_celula": int(row.cell_id),
                    "classe_bethesda": row.bethesda_system,
                    "rotulo_binario": label,
                    "alvo": int(row.alvo),
                    "nucleo_x": int(row.nucleus_x),
                    "nucleo_y": int(row.nucleus_y),
                    "caminho_imagem": row.caminho_imagem,
                    "caminho_recorte": str(crop_path),
                }
            )
    metadata = pd.DataFrame(rows)
    metadata.to_csv(config.metadata_csv, index=False)
    return metadata


def add_or_load_splits(config: Config, metadata: pd.DataFrame, force: bool = False) -> pd.DataFrame:
    if config.splits_csv.exists() and not force:
        print(f"Usando particoes existentes: {config.splits_csv}")
        return normalize_metadata_paths(config, pd.read_csv(config.splits_csv))

    table = metadata.copy().reset_index(drop=True)
    train_idx, temp_idx = next(
        GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=config.seed).split(
            table, groups=table["id_imagem"]
        )
    )
    temp = table.iloc[temp_idx]
    val_rel, test_rel = next(
        GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=config.seed).split(
            temp, groups=temp["id_imagem"]
        )
    )
    table["particao"] = "treino"
    table.loc[temp.index[val_rel], "particao"] = "validacao"
    table.loc[temp.index[test_rel], "particao"] = "teste"
    table.to_csv(config.splits_csv, index=False)
    return table


def prepare_data(config: Config, force_crops: bool = False, force_splits: bool = False) -> pd.DataFrame:
    metadata = generate_crops(config, force=force_crops)
    metadata = add_or_load_splits(config, metadata, force=force_splits)
    print(pd.crosstab(metadata["particao"], metadata["rotulo_binario"], margins=True))
    print(metadata.groupby("particao")["id_imagem"].nunique().rename("imagens"))
    return metadata


def load_prepared_metadata(config: Config) -> pd.DataFrame:
    if not config.splits_csv.exists():
        return prepare_data(config)
    return normalize_metadata_paths(config, pd.read_csv(config.splits_csv))
