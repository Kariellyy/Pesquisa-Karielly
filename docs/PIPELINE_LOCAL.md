# Pipeline local CRIC na RTX 3060 6 GB

Este repositório não depende mais de Jupyter para executar a pesquisa. O fluxo
local roda por uma CLI Python retomável, com checkpoints e CSVs gravados por
etapa.

## Hardware alvo

- NVIDIA GeForce RTX 3060 Laptop GPU
- 6 GB de VRAM
- Batch padrão: 48
- Precisão mista CUDA ativada

## Comandos principais

Verificar GPU e ambiente:

```powershell
.\run.ps1 check
```

Ver artefatos já existentes e o que ainda falta:

```powershell
.\run.ps1 status
```

Testar batches antes de uma rodada longa:

```powershell
.\run.ps1 benchmark -Batches 24,48,64 -Steps 20
```

Preparar recortes e partições:

```powershell
.\run.ps1 prepare
```

Treinar ConvNeXt-Tiny:

```powershell
.\run.ps1 train
```

Avaliar a partição retida com TTA, temperatura, limiares e bootstrap:

```powershell
.\run.ps1 eval
```

Rodar validação cruzada agrupada por imagem:

```powershell
.\run.ps1 cv
```

Treinar e avaliar baseline ResNet50:

```powershell
.\run.ps1 baseline
```

Rodar tudo:

```powershell
.\run.ps1 all
```

Gerar tabelas, métricas e figuras para o artigo:

```powershell
.\run.ps1 materials
```

Monitorar GPU em outro terminal:

```powershell
.\run.ps1 gpu
```

Forçar recomputação de uma etapa:

```powershell
.\run.ps1 train -Force
.\run.ps1 cv -Force
```

## Retomada e proteção contra perda

- `outputs_binary/checkpoints/best_convnext_tiny_binary.pt`: melhor ConvNeXt.
- `outputs_binary/checkpoints/best_convnext_tiny_binary.history.csv`: histórico do treino principal.
- `outputs_binary/checkpoints/best_resnet50_binary.pt`: baseline.
- `outputs_binary/checkpoints/cv_fold_*_convnext_tiny.pt`: folds da CV.
- `outputs_binary/metrics/validacao_cruzada_folds.csv`: salvo ao fim de cada fold.
- `outputs_binary/metrics/validacao_cruzada_resumo.csv`: média e desvio dos folds.
- `outputs_binary/metrics/comparacao_baseline.csv`: ConvNeXt x ResNet50.

Se uma etapa já possui artefato final, ela é pulada automaticamente. Use
`-Force` apenas quando quiser retreinar.

## Observação sobre CPU alta

Leitura de PNG, Pillow e aumentos de dados rodam na CPU. O modelo, o forward,
o backward e o otimizador rodam na GPU. A GPU pode aparecer abaixo de 100% se
a CPU estiver alimentando lotes mais devagar do que a GPU consome.

O lote 48 usa melhor a VRAM que o lote 24 original. O lote 64 pode ser mais
rápido, mas deve ser confirmado com `benchmark` antes de uma CV longa.
