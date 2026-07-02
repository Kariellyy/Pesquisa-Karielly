| Parametro | Valor |
| --- | --- |
| Arquitetura principal | ConvNeXt-Tiny |
| Arquitetura comparada | ResNet-50 |
| Dimensao de entrada | 224x224x3 |
| Batch size | 48 |
| Otimizador | AdamW |
| Taxa de aprendizado | 0.0003 |
| Decaimento de peso | 0.0001 |
| Epocas por fold | 6 |
| Early stopping | paciencia 2, delta 0.0005 |
| Label smoothing | 0.03 |
| Folds da CV | 5 |
| Agrupamento dos folds | id_imagem |
| Limiar por fold | 0,50 |
