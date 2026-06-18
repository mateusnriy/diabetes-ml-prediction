# Predição de Diabetes com Machine Learning

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status: Concluído](https://img.shields.io/badge/Status-Conclu%C3%ADdo-brightgreen.svg)]()

Este repositório contém o código, a documentação e os resultados do trabalho acadêmico de reprodução do artigo científico **"A comparison of machine learning algorithms for diabetes prediction"** (Khanam & Foo, 2021). O principal objetivo é comparar o desempenho de 7 classificadores clássicos de Machine Learning e 3 arquiteturas de Redes Neurais para o rastreio precoce do risco de diabetes em pacientes.

Utilizamos o dataset **Pima Indians Diabetes Database** da UCI, aplicando um pipeline rigoroso de pré-processamento de dados médicos. Isso incluiu a substituição de valores clínicos nulos representados por zeros, imputação pela mediana e remoção de outliers pela técnica IQR. Esse processo reduziu as amostras originais de 768 para 699 registros.

As 5 características mais correlacionadas com a classe de saída foram selecionadas via correlação de Pearson: `Glucose`, `BMI`, `Insulin`, `Pregnancies` e `Age`. Em seguida, dividimos o dataset de forma estratificada para os testes com 85% para treino e 15% para teste, e avaliamos os modelos utilizando validação K-fold (com k=7) e divisão Train/Test Split.

---

## 🔬 Referência do Artigo Base

> KHANAM, Jobeda Jamal; FOO, Simon Y. A comparison of machine learning algorithms for diabetes prediction. **ICT Express**, v. 7, n. 3, p. 432-439, 2021.  
> **DOI:** [10.1016/j.icte.2021.02.004](https://doi.org/10.1016/j.icte.2021.02.004)

---

## 📊 Resultados Obtidos

Os resultados das acurácias obtidas nos nossos testes estão compilados abaixo (o melhor modelo de rede neural e o melhor clássico estão em destaque):

| Algoritmo | Acurácia K-fold (Obtida) | Acurácia Split (Obtida) | Acurácia Split (Artigo) |
|---|---|---|---|
| Decision Tree | 62.94% | 68.42% | 73.14% |
| KNN (k=7) | 70.17% | 66.67% | 79.42% |
| Random Forest | 68.54% | 73.68% | 77.14% |
| Naive Bayes | 73.85% | 78.95% | 78.28% |
| AdaBoost | 71.47% | 71.93% | 79.42% |
| Logistic Regression | 75.46% | 78.95% | 78.85% |
| **SVM (Melhor Clássico)** | **73.59%** | **80.70%** | **77.71%** |
| NN-1 (200 épocas) | - | 63.16% | 83.81% |
| NN-1 (400 épocas) | - | 73.68% | 84.76% |
| NN-1 (800 épocas) | - | 77.19% | 82.86% |
| NN-2 (200 épocas) | - | 73.68% | 87.62% |
| **NN-2 (400 épocas - Melhor NN)** | **-** | **78.95%** | **88.57%** |
| NN-2 (800 épocas) | - | 78.95% | 87.62% |
| NN-3 (200 épocas) | - | 77.19% | 82.86% |
| NN-3 (400 épocas) | - | 75.44% | 83.81% |
| NN-3 (800 épocas) | - | 75.44% | 79.05% |

As análises gráficas e discussões completas sobre as pequenas discrepâncias causadas por evolução de bibliotecas de redes neurais (2021 para 2026) estão documentadas em nosso [Relatório Técnico](reports/relatorio_tecnico.md).

---

## 🛠️ Instalação e Execução

### Pré-requisitos
Certifique-se de ter instalado em sua máquina o **Python 3.10+** e o **Git**.

### 1. Clonar o Repositório
```bash
git clone https://github.com/mateusnriy/diabetes-ml-prediction.git
cd diabetes-ml-prediction
```

### 2. Configurar o Ambiente Virtual e Dependências
```bash
# Criar o ambiente virtual
python -m venv .venv

# Ativar o ambiente virtual (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Instalar dependências
pip install -r requirements.txt
```

### 3. Obter o Dataset
Conforme boas práticas descritas em [.gitignore](.gitignore), o arquivo de dados `diabetes.csv` não é versionado. 
Acesse o link no [data/README.md](data/README.md) para fazer o download do arquivo `diabetes.csv` do Kaggle e coloque-o na pasta `data/`.

### 4. Executar os Experimentos
Você pode executar o pipeline completo de forma automatizada por linha de comando rodando:
```bash
python -m ipykernel install --user --name=python3
python .venv\Scripts\python.exe -c "import subprocess; subprocess.run(['python', '.venv/Scripts/jupyter', 'nbconvert', '--to', 'notebook', '--execute', '--inplace', 'notebooks/*.ipynb'])"
```
Ou alternativamente, rode o script automatizado da pasta de testes:
```bash
python .venv\Scripts\execute_all.py
```
Isso gerará os resultados em `results/resultados.csv` e as imagens dos gráficos em `results/graficos/`.

---

## 📂 Estrutura do Repositório

```
diabetes-ml-prediction/
│
├── AGENTS.md                          # Instruções para o agente agy
├── README.md                          # Documentação principal
├── LICENSE                            # Licença MIT
├── .gitignore                         # Arquivos a ignorar no Git
├── requirements.txt                   # Dependências do projeto
│
├── docs/                              # Pasta de documentação técnica
│   ├── 00_AGENTS.md
│   ├── 01_requisitos.md
│   ├── 02_arquitetura.md
│   ├── ...
│
├── data/                              # Pasta com o dataset (local apenas)
│   └── README.md                      # Instruções de obtenção do dataset
│
├── notebooks/                         # Análise e modelagem
│   ├── 01_eda.ipynb
│   ├── 02_preprocessamento.ipynb
│   ├── 03_modelos_classicos.ipynb
│   └── 04_rede_neural.ipynb
│
├── src/                               # Módulos Python reutilizáveis
│   ├── __init__.py
│   ├── config.py                      # Constantes do sistema
│   ├── data_loader.py                 # Carregamento e estatísticas
│   ├── preprocessor.py                # Pipeline de pré-processamento
│   ├── models.py                      # Modelos clássicos de ML
│   ├── neural_network.py              # Redes neurais no Keras
│   ├── evaluator.py                   # Métricas e persistência
│   └── visualizer.py                  # Geração de gráficos
│
├── .agents/                           # Customizações adicionais
│   └── skills/                        # Skills em Markdown do agy
│
├── results/                           # Métricas e gráficos gerados
│   ├── resultados.csv
│   └── graficos/
│       ├── 01_acuracia_comparativa.png
│       ├── ...
│
└── reports/                           # Relatório científico final
    └── relatorio_tecnico.md
```

---

## 👥 Integrantes do Grupo

*   **Mateus Neri** - Desenvolvedor Principal / Reprodutor do Artigo
*   **Integrante 2** - Assistente de Pré-processamento
*   **Integrante 3** - Coletor de Métricas e Resultados
