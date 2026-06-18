# Dados — Pima Indians Diabetes Dataset

Este diretório deve conter o arquivo `diabetes.csv`.
O arquivo NÃO está incluído no repositório por boas práticas.

## Como Baixar

### Opção 1 — Kaggle (recomendado)
1. Acesse: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
2. Faça login (conta gratuita)
3. Clique em "Download"
4. Extraia o arquivo `diabetes.csv` para esta pasta

### Opção 2 — UCI Machine Learning Repository
- URL: https://archive.ics.uci.edu/dataset/34/diabetes

### Opção 3 — Via Kaggle API
```bash
pip install kaggle
kaggle datasets download -d uciml/pima-indians-diabetes-database
unzip pima-indians-diabetes-database.zip -d data/
```

## Descrição do Dataset

| Propriedade       | Valor                                      |
|-------------------|--------------------------------------------|
| Registros         | 768                                        |
| Atributos         | 9 (8 features + 1 target)                  |
| Target            | Outcome (0 = sem diabetes, 1 = com diabetes)|
| Fonte original    | NIDDK — National Institute of Diabetes     |
| Repositório       | UCI Machine Learning Repository            |
