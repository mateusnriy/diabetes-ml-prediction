# 03 — Estrutura de Pastas do Repositório

> O Antigravity CLI deve criar exatamente esta estrutura ao inicializar o projeto.
> Arquivos marcados com `[GERAR]` devem ter conteúdo gerado pelo agy.
> Arquivos marcados com `[MANUAL]` requerem intervenção humana.
> Arquivos marcados com `[AUTO]` são gerados automaticamente na execução.

---

## Árvore Completa

```
diabetes-ml-prediction/
│
├── AGENTS.md                          [GERAR] instruções para o agy
├── README.md                          [GERAR] documentação principal do projeto
├── LICENSE                            [GERAR] licença MIT
├── .gitignore                         [GERAR] arquivos a ignorar
├── requirements.txt                   [GERAR] dependências com versões exatas
│
├── docs/                              pasta de documentação técnica
│   ├── 00_AGENTS.md                   [MANUAL] este arquivo (já criado)
│   ├── 01_requisitos.md               [MANUAL] requisitos funcionais e não funcionais
│   ├── 02_arquitetura.md              [MANUAL] arquitetura e módulos
│   ├── 03_estrutura_pastas.md         [MANUAL] este arquivo
│   ├── 04_convencoes.md               [MANUAL] convenções de código e commits
│   ├── 05_dados.md                    [MANUAL] especificação do dataset
│   ├── 06_modelos.md                  [MANUAL] especificação dos modelos
│   ├── 07_experimentos.md             [MANUAL] plano de experimentos
│   ├── 08_relatorio.md                [MANUAL] estrutura do relatório técnico
│   └── 09_fluxo_trabalho.md           [MANUAL] fluxo de trabalho com o agy
│
├── data/
│   └── README.md                      [GERAR] instruções para baixar o dataset
│   (diabetes.csv NÃO é versionado — ver .gitignore)
│
├── notebooks/
│   ├── 01_eda.ipynb                   [GERAR] análise exploratória de dados
│   ├── 02_preprocessamento.ipynb      [GERAR] pipeline de pré-processamento
│   ├── 03_modelos_classicos.ipynb     [GERAR] 7 modelos clássicos de ML
│   └── 04_rede_neural.ipynb           [GERAR] 3 arquiteturas de Rede Neural
│
├── src/
│   ├── __init__.py                    [GERAR] vazio, torna src um pacote Python
│   ├── config.py                      [GERAR] constantes e configurações globais
│   ├── data_loader.py                 [GERAR] carregamento e validação do dataset
│   ├── preprocessor.py                [GERAR] pipeline de pré-processamento
│   ├── models.py                      [GERAR] modelos clássicos de ML
│   ├── neural_network.py              [GERAR] redes neurais com Keras
│   ├── evaluator.py                   [GERAR] métricas e persistência
│   └── visualizer.py                  [GERAR] geração de gráficos
│
├── .agents/
│   └── skills/
│       ├── preprocessing.md           [GERAR] skill de pré-processamento para o agy
│       ├── modeling.md                [GERAR] skill de modelagem para o agy
│       └── reporting.md               [GERAR] skill de relatório para o agy
│
├── results/
│   ├── resultados.csv                 [AUTO] métricas de todos os modelos
│   └── graficos/
│       ├── 01_acuracia_comparativa.png   [AUTO]
│       ├── 02_f1_comparativo.png         [AUTO]
│       ├── 03_matriz_confusao_nn.png     [AUTO]
│       ├── 04_curva_roc.png              [AUTO]
│       ├── 05_learning_curves_nn.png     [AUTO]
│       ├── 06_feature_importance.png     [AUTO]
│       └── 07_comparacao_artigo.png      [AUTO]
│
└── reports/
    └── relatorio_tecnico.md           [GERAR] relatório técnico do trabalho
```

---

## Conteúdo Obrigatório de Cada Arquivo-Base

### `.gitignore`

```gitignore
# Dados (nunca versionar)
data/*.csv
data/*.xlsx
data/*.json

# Python
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
*.egg-info/
dist/
build/

# Jupyter
.ipynb_checkpoints/
*.ipynb_checkpoints

# Ambientes
.env
.venv/
venv/
env/

# Resultados gerados (opcional — manter ou ignorar)
# results/graficos/*.png
# results/*.csv

# IDEs
.vscode/
.idea/
*.swp

# macOS
.DS_Store
```

---

### `requirements.txt`

```txt
# Manipulação de dados
pandas==2.2.2
numpy==1.26.4

# Machine Learning
scikit-learn==1.4.2
imbalanced-learn==0.12.3

# Deep Learning
tensorflow==2.16.1

# Visualização
matplotlib==3.8.4
seaborn==0.13.2

# IA Explicável
shap==0.45.1

# Notebooks
jupyter==1.0.0
ipykernel==6.29.4

# Utilitários
pathlib2==2.3.7
```

---

### `data/README.md`

```markdown
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
```

---

### `results/graficos/.gitkeep`

Arquivo vazio para versionar a pasta no Git mesmo sem gráficos gerados:

```
(arquivo vazio)
```

---

## Criação da Estrutura com o agy

Use este prompt no Antigravity CLI para criar toda a estrutura:

```
Leia o arquivo docs/03_estrutura_pastas.md e crie toda a estrutura de
diretórios e arquivos-base do projeto conforme especificado. Para cada
arquivo marcado com [GERAR], crie o arquivo com o conteúdo exato indicado
no documento. Para pastas sem arquivos, crie um .gitkeep. Não crie
nenhum arquivo de dados (.csv).
```