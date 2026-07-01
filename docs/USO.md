# Uso da pipeline

Este projeto roda por uma CLI Python chamada pelo script `run.ps1`. O script
define `PYTHONPATH=src` e executa `python -m cric_pipeline`.

## Preparação do ambiente

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-local.txt
```

## Comandos

```powershell
.\run.ps1 check
.\run.ps1 status
.\run.ps1 prepare
.\run.ps1 train
.\run.ps1 eval
.\run.ps1 cv
.\run.ps1 baseline
.\run.ps1 materials
```

Use `.\run.ps1 all` para executar o fluxo completo. Etapas com artefatos finais
já existentes são reaproveitadas; use `-Force` apenas quando quiser recomputar.

## Saídas principais

- `outputs_binary/metadata_binary.csv`: metadados dos recortes.
- `outputs_binary/metadata_splits.csv`: partições por imagem.
- `outputs_binary/checkpoints/`: checkpoints treinados.
- `outputs_binary/metrics/`: métricas do teste, validação cruzada e baseline.
- `materiais_artigo/`: tabelas, figuras e resumos prontos para o texto.
