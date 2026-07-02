# Triagem Binária de Células Cervicais em Esfregaços Convencionais com ConvNeXt-Tiny e Particionamento por Imagem

**Karielly de Carvalho**  
Universidade Federal do Piauí (UFPI), Campus Senador Helvídio Nunes de Barros  
Orientadores: **João Antônio Leal de Miranda** e **Romuere Rodrigues Veloso e Silva**

## Resumo

Este repositório reúne o código, a documentação experimental e os materiais de
apoio do estudo sobre triagem binária de células cervicais em esfregaços
convencionais de Papanicolau. O trabalho utiliza a base pública CRIC Cervix e
avalia a distinção entre células `normal` e `anormal` por meio de redes neurais
convolucionais profundas.

O modelo principal do estudo é a ConvNeXt-Tiny. Como comparação complementar,
uma ResNet-50 é treinada sob o mesmo protocolo experimental. A validação
principal é conduzida por validação cruzada em cinco folds, com agrupamento por
imagem-fonte, evitando que células provenientes da mesma imagem sejam usadas
simultaneamente no treinamento e na avaliação de um mesmo fold.

**Palavras-chave:** citologia cervical; CRIC Cervix; ConvNeXt-Tiny; ResNet-50;
validação cruzada agrupada; triagem binária.

## Delineamento Experimental

A tarefa foi formulada como classificação binária de recortes celulares. As
classes ASC-US, LSIL, ASC-H, HSIL e SCC foram agrupadas como `anormal`,
enquanto a classe Negative for intraepithelial lesion foi mantida como
`normal`.

O protocolo válido para o artigo considera:

- validação cruzada em 5 folds;
- separação agrupada por `id_imagem`;
- mesma definição de folds para ConvNeXt-Tiny e ResNet-50;
- validação interna dentro de cada fold para seleção do checkpoint;
- limiar fixo de decisão em `0,50` para métricas dependentes de limiar;
- AUC ROC e AUC PR como métricas principais.

Partições fixas de treino, validação e teste não constituem o eixo principal da
pesquisa. O desenho experimental central é a validação cruzada agrupada.

## Base de Dados

A base utilizada é a CRIC Cervix, disponibilizada publicamente no Figshare. A
pipeline deste repositório baixa e organiza os arquivos no formato necessário
para os experimentos:

- `cric_cervix/classification/classifications.csv`;
- `cric_cervix/classification/classifications.json`;
- `cric_cervix/classification/README.md`;
- `cric_cervix/images/cric_image_001_<hash>.png` até `cric_image_400_<hash>.png`.

Os dados brutos, recortes, checkpoints e arquivos compactados gerados localmente
não devem ser versionados no Git.

## Reprodutibilidade

O ambiente local esperado é controlado por `requirements-local.txt` e
`configs/local_3060.json`. O fluxo completo pode ser executado por:

```powershell
.\run.ps1 check
.\run.ps1 status
.\run.ps1 download-data
.\run.ps1 prepare
.\run.ps1 cv-convnext
.\run.ps1 cv-resnet50
.\run.ps1 compare-cv
.\run.ps1 materials
```

Para validar o downloader sem baixar todas as imagens:

```powershell
.\run.ps1 download-data -LimitImages 1
```

Se os folds da ConvNeXt-Tiny já estiverem treinados, a comparação com a
ResNet-50 pode ser atualizada com:

```powershell
.\run.ps1 cv-resnet50
.\run.ps1 compare-cv
.\run.ps1 materials
```

O uso de `-Force` deve ser reservado para recomputar uma etapa já existente.

## Métricas

Os resultados são avaliados por AUC ROC, AUC PR, sensibilidade,
especificidade, acurácia balanceada, F1 da classe anormal, coeficiente de
correlação de Matthews, kappa de Cohen e Brier score. A comparação entre
ConvNeXt-Tiny e ResNet-50 é feita fold a fold, usando os mesmos grupos de
holdout para reduzir variação atribuída ao particionamento.

## Organização do Repositório

- `artigo/`: fonte LaTeX, figuras, dados e PDF do artigo.
- `src/cric_pipeline/`: implementação da pipeline experimental.
- `docs/PIPELINE_LOCAL.md`: instruções detalhadas de execução.
- `configs/local_3060.json`: configuração local dos experimentos.
- `materiais_artigo/`: tabelas, figuras e métricas consolidadas.
- `cric_cervix/`: base CRIC baixada localmente, ignorada pelo Git.
- `outputs_binary/`: recortes, checkpoints, métricas e comparações, ignorados pelo Git.

## Artefatos de Saída

- `outputs_binary/cv/fold_assignments.csv`: definição dos folds por imagem.
- `outputs_binary/convnext_tiny/metrics/validacao_cruzada_folds.csv`: métricas por fold da ConvNeXt-Tiny.
- `outputs_binary/resnet50/metrics/validacao_cruzada_folds.csv`: métricas por fold da ResNet-50.
- `outputs_binary/comparacao_cv/comparacao_cv_pareada.csv`: deltas fold a fold entre os modelos.
- `materiais_artigo/`: materiais exportados para redação e apresentação do trabalho.
