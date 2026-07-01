# Pesquisa CRIC ConvNeXt-Tiny

Repositório acadêmico da pesquisa **Triagem Binária de Células Cervicais em
Esfregaços Convencionais com ConvNeXt-Tiny e Particionamento por Imagem**.

- Autora: Karielly de Carvalho
- Coorientador: João Antônio Leal de Miranda
- Orientador: Romuere Rodrigues Veloso e Silva
- Instituição: Universidade Federal do Piauí (UFPI), Campus Senador Helvídio
  Nunes de Barros (CSHNB)
- Repositório: <https://github.com/Kariellyy/Pesquisa-Karielly>

O trabalho avalia uma pipeline de triagem binária normal/anormal de células
cervicais da coleção pública CRIC Cervix, com ConvNeXt-Tiny, partições
agrupadas por imagem-fonte e análise de limiares operacionais.

## Estrutura

- `src/cric_pipeline/`: código-fonte da pipeline.
- `configs/local_3060.json`: configuração dos experimentos locais.
- `scripts/`: scripts auxiliares de verificação.
- `docs/`: documentação de uso, dados, resultados e execução local.
- `materiais_artigo/`: tabelas, métricas e figuras finais geradas para o artigo.
- `artigo - romuere/`: artigo compacto para entrega/apresentação.
- `artigo - eniac/`: artigo completo preparado para submissão ao ENIAC.
- `cric_cervix/`: base CRIC local, não versionada.
- `outputs_binary/`: recortes, checkpoints, métricas e saídas intermediárias,
  não versionados.

## Instalação

O fluxo local foi preparado para Windows/PowerShell com Python 3.12 e GPU NVIDIA
RTX 3060 Laptop de 6 GB.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-local.txt
```

Organize a base CRIC em `cric_cervix/`, conforme descrito em
[`docs/DADOS.md`](docs/DADOS.md).

## Execução

Verificar ambiente e GPU:

```powershell
.\run.ps1 check
```

Executar etapas individuais:

```powershell
.\run.ps1 status
.\run.ps1 prepare
.\run.ps1 benchmark
.\run.ps1 train
.\run.ps1 eval
.\run.ps1 cv
.\run.ps1 baseline
.\run.ps1 materials
```

Executar tudo em sequência:

```powershell
.\run.ps1 all
```

Monitorar GPU:

```powershell
.\run.ps1 gpu
```

Documentação detalhada: [`docs/PIPELINE_LOCAL.md`](docs/PIPELINE_LOCAL.md) e
[`docs/USO.md`](docs/USO.md).

## Resultados e artigo

Os materiais finais usados no texto ficam em `materiais_artigo/`. A base
completa e os checkpoints treinados não são versionados por tamanho; eles devem
ser regenerados localmente ou mantidos fora do Git.

O artigo compacto para entrega está em `artigo - romuere/`. O artigo completo do
ENIAC está em `artigo - eniac/`.

## Aviso acadêmico

Este repositório tem finalidade acadêmica e reprodutível. O método descrito é
um protótipo de pesquisa para priorização e não possui finalidade diagnóstica.
