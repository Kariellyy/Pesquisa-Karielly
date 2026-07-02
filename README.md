# Pesquisa CRIC - comparacao K-fold

Pipeline local para comparar ConvNeXt-Tiny e ResNet-50 na classificacao binaria
normal/anormal de celulas cervicais da base CRIC.

O fluxo valido para o artigo e a validacao cruzada agrupada por imagem, com 5
folds e mesma definicao de holdout por fold para as duas arquiteturas. A CV usa
ate 6 epocas por fold com early stopping, pois os folds ja treinados do
ConvNeXt-Tiny atingiram melhor validacao entre as epocas 1 e 6. O comparativo
antigo por treino/validacao/teste fixos foi arquivado e nao deve ser usado no
paper.

## Comandos principais

```powershell
.\run.ps1 check
.\run.ps1 status
.\run.ps1 prepare
.\run.ps1 cv-convnext
.\run.ps1 cv-resnet50
.\run.ps1 compare-cv
.\run.ps1 materials
```

Como o ConvNeXt-Tiny ja possui os 5 folds treinados, o comando novo mais
importante e:

```powershell
.\run.ps1 cv-resnet50
.\run.ps1 compare-cv
.\run.ps1 materials
```

## Pastas principais

- `src/cric_pipeline/`: codigo da pipeline.
- `configs/local_3060.json`: configuracao local da RTX 3060.
- `cric_cervix/`: base de dados local, ignorada pelo Git.
- `outputs_binary/convnext_tiny/`: folds, historicos e metricas do ConvNeXt-Tiny.
- `outputs_binary/resnet50/`: folds, historicos e metricas do ResNet-50.
- `outputs_binary/comparacao_cv/`: comparacao fold a fold entre os modelos.
- `archive/`: notebooks e resultados antigos preservados como historico.

Documentacao detalhada: [docs/PIPELINE_LOCAL.md](docs/PIPELINE_LOCAL.md).
