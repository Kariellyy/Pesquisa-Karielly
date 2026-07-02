from __future__ import annotations

import hashlib
import json
import re
import shutil
import time
from pathlib import Path
from urllib.request import Request, urlopen

from .config import Config


FIGSHARE_API = "https://api.figshare.com/v2"
CRIC_COLLECTION_ID = 4960286
CRIC_CLASSIFICATION_ARTICLE_ID = 12233156
USER_AGENT = "Pesquisa-Karielly-CRIC-downloader/1.0"


def fetch_json(url: str) -> object:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def download_url(url: str, target: Path, force: bool = False) -> None:
    if target.exists() and not force:
        return

    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".download")
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=120) as response, tmp.open("wb") as out:
        shutil.copyfileobj(response, out)
    tmp.replace(target)


def file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def expected_md5(file_info: dict) -> str:
    return str(file_info.get("computed_md5") or file_info.get("supplied_md5") or "")


def ensure_file(file_info: dict, target: Path, force: bool = False) -> bool:
    md5 = expected_md5(file_info)
    if target.exists() and not force:
        if not md5 or file_md5(target) == md5:
            return False
        print(f"MD5 diferente; baixando novamente: {target}")

    download_url(str(file_info["download_url"]), target, force=True)
    if md5:
        got = file_md5(target)
        if got != md5:
            target.unlink(missing_ok=True)
            raise RuntimeError(f"MD5 invalido para {target}: esperado {md5}, obtido {got}")
    return True


def iter_collection_articles(collection_id: int = CRIC_COLLECTION_ID) -> list[dict]:
    articles: list[dict] = []
    page = 1
    while True:
        url = f"{FIGSHARE_API}/collections/{collection_id}/articles?page_size=100&page={page}"
        chunk = fetch_json(url)
        if not isinstance(chunk, list):
            raise RuntimeError(f"Resposta inesperada do Figshare: {url}")
        if not chunk:
            break
        articles.extend(chunk)
        page += 1
    return articles


def image_number(article: dict) -> int:
    title = str(article.get("title", ""))
    match = re.search(r"#(\d+)", title)
    if not match:
        raise RuntimeError(f"Nao foi possivel identificar numero da imagem: {title}")
    return int(match.group(1))


def article_files(article_id: int) -> list[dict]:
    files = fetch_json(f"{FIGSHARE_API}/articles/{article_id}/files")
    if not isinstance(files, list):
        raise RuntimeError(f"Resposta inesperada nos arquivos do artigo {article_id}")
    return files


def download_classifications(config: Config, force: bool = False) -> int:
    files = article_files(CRIC_CLASSIFICATION_ARTICLE_ID)
    wanted = {"classifications.csv", "classifications.json", "README.md"}
    count = 0
    for file_info in files:
        name = str(file_info["name"])
        if name not in wanted:
            continue
        target = config.data_dir / "classification" / name
        if ensure_file(file_info, target, force=force):
            count += 1
            print(f"Baixado: {target}")
    return count


def download_images(config: Config, force: bool = False, limit: int | None = None) -> int:
    articles = [
        item
        for item in iter_collection_articles()
        if int(item["id"]) != CRIC_CLASSIFICATION_ARTICLE_ID
    ]
    articles = sorted(articles, key=image_number)
    if limit is not None:
        articles = articles[:limit]

    count = 0
    for index, article in enumerate(articles, start=1):
        number = image_number(article)
        files = [
            file_info
            for file_info in article_files(int(article["id"]))
            if str(file_info["name"]).lower().endswith(".png")
        ]
        if len(files) != 1:
            raise RuntimeError(f"Esperado 1 PNG no artigo {article['id']}; encontrados {len(files)}")

        file_info = files[0]
        target = config.images_dir / f"cric_image_{number:03d}_{file_info['name']}"
        if ensure_file(file_info, target, force=force):
            count += 1
            print(f"Baixada imagem {number:03d}: {target.name}")
        else:
            print(f"Imagem {number:03d} ja existe: {target.name}")
        time.sleep(0.05)
        if index % 25 == 0:
            print(f"Progresso imagens: {index}/{len(articles)}")
    return count


def download_cric_dataset(config: Config, force: bool = False, limit_images: int | None = None) -> None:
    config.data_dir.mkdir(parents=True, exist_ok=True)
    print(f"Destino da CRIC: {config.data_dir}")
    class_count = download_classifications(config, force=force)
    image_count = download_images(config, force=force, limit=limit_images)
    expected = limit_images if limit_images is not None else 400
    available = len(list(config.images_dir.glob("cric_image_*.png")))
    print(f"Arquivos de classificacao baixados/atualizados: {class_count}")
    print(f"Imagens baixadas/atualizadas: {image_count}")
    print(f"Imagens disponiveis em {config.images_dir}: {available}/{expected}")
    if limit_images is None and available != 400:
        raise RuntimeError(f"Download incompleto da CRIC: esperado 400 imagens, encontrado {available}")


def raw_dataset_available(config: Config) -> bool:
    return config.annotations_csv.exists() and len(list(config.images_dir.glob("cric_image_*.png"))) == 400
