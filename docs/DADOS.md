# Dados

O estudo utiliza a coleção pública CRIC Cervix citada no artigo. Os dados brutos
não ficam versionados neste repositório.

## Estrutura esperada

Coloque a base localmente em `cric_cervix/`:

```text
cric_cervix/
  images/
  classification/
    classifications.csv
```

A pasta `cric_cervix/` está no `.gitignore` para evitar publicar dados grandes
ou cópias locais da base.

## Caminhos

O arquivo `configs/local_3060.json` define:

- `data_dir`: `cric_cervix`
- `output_dir`: `outputs_binary`

Evite caminhos absolutos. A pipeline resolve esses diretórios a partir da raiz
do repositório.
