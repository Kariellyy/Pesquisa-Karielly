# Resultados

Os resultados finais usados no artigo estão em `materiais_artigo/`. Essa pasta
é gerada a partir de `outputs_binary/` com:

```powershell
.\run.ps1 materials
```

## Conteúdo

- `materiais_artigo/figuras/`: figuras exportadas para análise e apresentação.
- `materiais_artigo/tabelas/`: tabelas em CSV, Markdown e LaTeX.
- `materiais_artigo/metricas/`: resumo de métricas em Markdown e JSON.
- `materiais_artigo/dados_origem/`: CSVs de origem usados para gerar os
  materiais finais.

## Reprodutibilidade

Se o treinamento for reexecutado com `-Force`, os valores podem mudar conforme a
configuração, semente, hardware e versões das dependências. Depois de retreinar,
gere novamente `materiais_artigo/` e confira se o artigo continua consistente
com os novos arquivos.
