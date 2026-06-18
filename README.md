# Predição de Diabetes com Machine Learning

[![Python Version](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Resumo

Este projeto reproduz e valida experimentalmente os resultados do artigo **"A comparison of machine learning algorithms for diabetes prediction"** (Khanam & Foo, 2021), publicado na *ICT Express*. A pesquisa compara o desempenho de **7 classificadores clássicos de Machine Learning** e **3 arquiteturas de Redes Neurais Artificiais** na tarefa de rastreio precoce de diabetes mellitus tipo 2.

O estudo utiliza o **Pima Indians Diabetes Database** (UCI), composto por 768 registros clínicos de mulheres de herança indígena Pima com 21 anos ou mais. Após o pipeline de pré-processamento — que inclui substituição de zeros biologicamente impossíveis por `NaN`, imputação pela mediana e remoção de outliers via IQR restrita a variáveis contínuas — o conjunto de dados foi reduzido a **699 registros válidos**.

A seleção de atributos, baseada em correlação de Pearson (limiar ≥ 0,20), reteve 5 variáveis preditoras: `Glucose`, `BMI`, `Insulin`, `Pregnancies` e `Age`. A avaliação seguiu duas estratégias: validação cruzada K-fold (k=7) e divisão estratificada treino/teste (85%/15%).

---

## Referência Bibliográfica

> KHANAM, Jobeda Jamal; FOO, Simon Y. A comparison of machine learning algorithms for diabetes prediction. **ICT Express**, v. 7, n. 3, p. 432–439, 2021.
> **DOI:** [10.1016/j.icte.2021.02.004](https://doi.org/10.1016/j.icte.2021.02.004)

---

## Resultados Experimentais

A tabela a seguir consolida as acurácias obtidas nesta reprodução frente aos valores reportados no artigo original. Os melhores modelos em cada categoria estão destacados.

| Algoritmo | Acurácia K-fold (Obtida) | Acurácia Split (Obtida) | Acurácia Split (Artigo) |
|---|---|---|---|
| Decision Tree | 62,94% | 68,42% | 73,14% |
| KNN (k=7) | 70,17% | 66,67% | 79,42% |
| Random Forest | 68,54% | 73,68% | 77,14% |
| Naive Bayes | 73,85% | 78,95% | 78,28% |
| AdaBoost | 71,47% | 71,93% | 79,42% |
| Logistic Regression | 75,46% | 78,95% | 78,85% |
| **SVM (Melhor Clássico)** | **73,59%** | **80,70%** | **77,71%** |
| NN-1 (200 épocas) | — | 63,16% | 83,81% |
| NN-1 (400 épocas) | — | 73,68% | 84,76% |
| NN-1 (800 épocas) | — | 77,19% | 82,86% |
| NN-2 (200 épocas) | — | 73,68% | 87,62% |
| **NN-2 (400 épocas — Melhor NN)** | **—** | **78,95%** | **88,57%** |
| NN-2 (800 épocas) | — | 78,95% | 87,62% |
| NN-3 (200 épocas) | — | 77,19% | 82,86% |
| NN-3 (400 épocas) | — | 75,44% | 83,81% |
| NN-3 (800 épocas) | — | 75,44% | 79,05% |

> **Nota metodológica:** As discrepâncias observadas nas redes neurais são atribuídas à evolução das bibliotecas TensorFlow/Keras entre 2021 e 2026, conforme discutido em detalhe no [Relatório Técnico](reports/relatorio_tecnico.md).

---

## Instalação e Execução

### Pré-requisitos

- **Python 3.12** (versão estável recomendada)
- **Git**
- Acesso ao dataset (instruções na Etapa 3)

### 1. Clonar o Repositório

```bash
git clone https://github.com/mateusnriy/diabetes-ml-prediction.git
cd diabetes-ml-prediction
```

### 2. Configurar o Ambiente Virtual e Dependências

```bash
# Criar o ambiente virtual com versão estável do Python
py -3.12 -m venv .venv

# Ativar o ambiente virtual (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Instalar todas as dependências
pip install -r requirements.txt
```

### 3. Obter o Dataset

O arquivo `diabetes.csv` não é versionado neste repositório, conforme boas práticas definidas no [.gitignore](.gitignore). Acesse as instruções em [data/README.md](data/README.md), faça o download a partir do Kaggle e coloque o arquivo na pasta `data/`.

### 4. Executar o Pipeline de Experimentos

A execução deve ser feita **de forma sequencial** através dos notebooks Jupyter, na ordem abaixo:

```bash
# Iniciar o servidor Jupyter
jupyter notebook
```

No navegador, abra e execute cada notebook na seguinte ordem:

| Ordem | Notebook | Descrição |
|-------|----------|-----------|
| 1º | `notebooks/01_eda.ipynb` | Análise Exploratória de Dados |
| 2º | `notebooks/02_preprocessamento.ipynb` | Pré-processamento e limpeza |
| 3º | `notebooks/03_modelos_classicos.ipynb` | Treinamento dos 7 classificadores |
| 4º | `notebooks/04_rede_neural.ipynb` | Treinamento das 3 arquiteturas de NN |

> **Importante:** Cada notebook é autocontido e pode ser executado individualmente do início ao fim (`Kernel → Restart & Run All`). Os resultados serão salvos automaticamente em `results/resultados.csv` e os gráficos em `results/graficos/`.

---

## Estrutura do Repositório

```
diabetes-ml-prediction/
│
├── AGENTS.md                          # Instruções para o agente agy
├── README.md                          # Documentação principal
├── LICENSE                            # Licença MIT
├── .gitignore                         # Arquivos a ignorar no Git
├── requirements.txt                   # Dependências do projeto
│
├── docs/                              # Documentação técnica do projeto
│   ├── 01_requisitos.md
│   ├── 02_arquitetura.md
│   ├── 03_estrutura_pastas.md
│   ├── 04_convencoes.md
│   ├── 05_dados.md
│   ├── 06_modelos.md
│   ├── 07_experimentos.md
│   ├── 08_relatorio.md
│   └── 09_fluxo_trabalho.md
│
├── data/                              # Dataset (não versionado)
│   └── README.md                      # Instruções de obtenção do dataset
│
├── notebooks/                         # Pipeline de análise e modelagem
│   ├── 01_eda.ipynb
│   ├── 02_preprocessamento.ipynb
│   ├── 03_modelos_classicos.ipynb
│   └── 04_rede_neural.ipynb
│
├── src/                               # Módulos Python reutilizáveis
│   ├── __init__.py
│   ├── config.py                      # Constantes e hiperparâmetros
│   ├── data_loader.py                 # Carregamento e validação
│   ├── preprocessor.py                # Pipeline de pré-processamento
│   ├── models.py                      # Modelos clássicos de ML
│   ├── neural_network.py              # Redes neurais (Keras/TensorFlow)
│   ├── evaluator.py                   # Métricas e persistência
│   └── visualizer.py                  # Geração de gráficos
│
├── results/                           # Saídas geradas automaticamente
│   ├── resultados.csv                 # Métricas consolidadas
│   └── graficos/                      # Visualizações exportadas
│
└── reports/                           # Relatório científico final
    └── relatorio_tecnico.md
```

---

## Integrantes do Grupo

- **Mateus Neri** — Desenvolvedor Principal / Reprodutor do Artigo
- **Juliana Assis** — Assistente de Pré-processamento
- **Fávio de Aguiar** — Coletor de Métricas e Resultados
