# Pipeline local CRIC com validacao cruzada

Este repositorio roda a comparacao binaria normal/anormal da CRIC por validacao
cruzada agrupada por imagem. O desenho experimental valido para o artigo e:

- 5 folds com `StratifiedGroupKFold`;
- grupo de separacao: `id_imagem`;
- mesmos folds para ConvNeXt-Tiny e ResNet-50;
- mesmos folds, aumentos, otimizador, batch size e limiar 0,50;
- ate 6 epocas por fold, com early stopping de paciencia 2;
- unica diferenca intencional: arquitetura do backbone.

O comparativo antigo por treino/validacao/teste fixos foi arquivado em `archive`
e nao deve ser citado como comparacao entre ConvNeXt-Tiny e ResNet-50.

Os historicos ja salvos do ConvNeXt-Tiny indicam melhor AUC de validacao nas
epocas 1, 3, 3, 4 e 6, respectivamente. Por isso, o teto de 6 epocas reduz o
tempo sem cortar a faixa em que o modelo principal convergiu.

## Hardware alvo

- NVIDIA GeForce RTX 3060 Laptop GPU
- 6 GB de VRAM
- Batch padrao: 48
- Precisao mista CUDA ativada

## Comandos

Verificar GPU e ambiente:

```powershell
.\run.ps1 check
```

Ver artefatos existentes:

```powershell
.\run.ps1 status
```

Baixar a base CRIC Cervix do Figshare:

```powershell
.\run.ps1 download-data
```

O downloader usa a API publica do Figshare. As classificacoes vem do artigo
`CRIC Cervix Classification` (`12233156`) e as 400 imagens vem da colecao
`4960286`. Os arquivos sao organizados no formato consumido pelo codigo:

- `cric_cervix/classification/classifications.csv`
- `cric_cervix/classification/classifications.json`
- `cric_cervix/classification/README.md`
- `cric_cervix/images/cric_image_001_<hash>.png` ate `cric_image_400_<hash>.png`

Para testar o downloader sem baixar a base inteira:

```powershell
.\run.ps1 download-data -LimitImages 1
```

Preparar recortes e metadados, se ainda faltar:

```powershell
.\run.ps1 prepare
```

Rodar ou retomar a CV do ConvNeXt-Tiny:

```powershell
.\run.ps1 cv-convnext
```

Rodar ou retomar a CV do ResNet-50 nos mesmos folds:

```powershell
.\run.ps1 cv-resnet50
```

Gerar a comparacao pareada fold a fold:

```powershell
.\run.ps1 compare-cv
```

Gerar tabelas, metricas e figuras para o artigo:

```powershell
.\run.ps1 materials
```

Executar o fluxo completo:

```powershell
.\run.ps1 all
```

Monitorar GPU em outro terminal:

```powershell
.\run.ps1 gpu
```

Forcar recomputacao de uma etapa:

```powershell
.\run.ps1 cv-resnet50 -Force
```

Use `-Force` com cuidado: ele retreina os folds da arquitetura chamada.

## Artefatos

- `outputs_binary/cv/fold_assignments.csv`: definicao unica dos folds por imagem.
- `outputs_binary/convnext_tiny/checkpoints/cv_fold_*_convnext_tiny.pt`: checkpoints ConvNeXt-Tiny.
- `outputs_binary/convnext_tiny/metrics/validacao_cruzada_folds.csv`: metricas por fold ConvNeXt-Tiny.
- `outputs_binary/convnext_tiny/metrics/validacao_cruzada_resumo.csv`: media e desvio ConvNeXt-Tiny.
- `outputs_binary/resnet50/checkpoints/cv_fold_*_resnet50.pt`: checkpoints ResNet-50.
- `outputs_binary/resnet50/metrics/validacao_cruzada_folds.csv`: metricas por fold ResNet-50.
- `outputs_binary/resnet50/metrics/validacao_cruzada_resumo.csv`: media e desvio ResNet-50.
- `outputs_binary/comparacao_cv/comparacao_cv_folds.csv`: folds dos dois modelos em formato longo.
- `outputs_binary/comparacao_cv/comparacao_cv_resumo.csv`: media e desvio por modelo/metrica.
- `outputs_binary/comparacao_cv/comparacao_cv_pareada.csv`: deltas fold a fold.

## Retomada

Cada fold salvo e reaproveitado automaticamente. Se o treino do ResNet-50 parar
no fold 3, por exemplo, rode novamente:

```powershell
.\run.ps1 cv-resnet50
```

A pipeline pula os folds ja presentes no CSV de metricas e continua do proximo.
