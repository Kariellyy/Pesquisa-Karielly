# Pesquisa CRIC ConvNeXt-Tiny

Pipeline local para triagem binária normal/anormal de células cervicais da CRIC
em uma NVIDIA RTX 3060 Laptop de 6 GB.

## Rodar

```powershell
.\run.ps1 check
.\run.ps1 status
.\run.ps1 prepare
.\run.ps1 benchmark
.\run.ps1 train
.\run.ps1 eval
.\run.ps1 cv
.\run.ps1 baseline
.\run.ps1 materials
```

Ou tudo em sequência:

```powershell
.\run.ps1 all
```

Monitorar GPU:

```powershell
.\run.ps1 gpu
```

Documentação detalhada: [docs/PIPELINE_LOCAL.md](docs/PIPELINE_LOCAL.md).

## Pastas principais

- `src/cric_pipeline/`: código da pipeline.
- `configs/local_3060.json`: configuração da RTX 3060.
- `cric_cervix/`: base de dados, preservada localmente.
- `outputs_binary/`: recortes, checkpoints, métricas e figuras.
- `artigo/`: artigo LaTeX, não usado pela pipeline.
- `archive/notebooks/`: notebooks antigos preservados como histórico.
