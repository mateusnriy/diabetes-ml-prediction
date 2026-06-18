# 02 — Arquitetura do Projeto

> O Antigravity CLI deve consultar este arquivo ao gerar módulos em `src/` ou ao criar a estrutura de notebooks.

---

## 1. Visão Geral da Arquitetura

O projeto segue uma arquitetura em camadas, separando responsabilidades entre notebooks (interface de análise), módulos Python (lógica reutilizável) e artefatos de resultado.

```
┌─────────────────────────────────────────────────────────┐
│                    CAMADA DE INTERFACE                  │
│              notebooks/ (Jupyter Notebooks)             │
│  01_eda  │  02_prep  │  03_modelos  │  04_rede_neural   │
└─────────────────────┬───────────────────────────────────┘
                      │ importa
┌─────────────────────▼───────────────────────────────────┐
│                   CAMADA DE LÓGICA                      │
│                      src/                               │
│  config.py │ data_loader.py │ preprocessor.py           │
│  models.py │ evaluator.py   │ visualizer.py             │
└─────────────────────┬───────────────────────────────────┘
                      │ lê / salva
┌─────────────────────▼───────────────────────────────────┐
│                   CAMADA DE DADOS                       │
│         data/          │        results/                │
│  diabetes.csv          │  resultados.csv                │
│  (não versionado)      │  graficos/*.png                │
└─────────────────────────────────────────────────────────┘
```

---

## 2. Fluxo de Dados

```
[diabetes.csv]
      │
      ▼
┌─────────────┐
│ data_loader │  → carrega, valida shape e tipos
└──────┬──────┘
       │
       ▼
┌──────────────┐
│ preprocessor │  → zeros→NaN → mediana → outliers → features → normalização → split
└──────┬───────┘
       │
       ├──────────────────────────────────────────┐
       ▼                                          ▼
┌─────────────┐                          ┌──────────────┐
│   models    │                          │ neural_net   │
│  (7 algos)  │                          │ (NN-1,2,3)   │
└──────┬──────┘                          └──────┬───────┘
       │                                        │
       └──────────────┬─────────────────────────┘
                      ▼
              ┌───────────────┐
              │   evaluator   │  → calcula métricas, salva resultados.csv
              └───────┬───────┘
                      │
                      ▼
              ┌───────────────┐
              │  visualizer   │  → gera e salva todos os gráficos PNG
              └───────────────┘
```

---

## 3. Módulos de `src/`

### `src/config.py`
Centraliza todas as constantes do projeto. **Nenhum outro módulo deve ter valores hard-coded.**

```python
"""
config.py — Constantes e configurações globais do projeto.
"""
from pathlib import Path

# ── Caminhos ────────────────────────────────────────────
ROOT_DIR    = Path(__file__).parent.parent
DATA_DIR    = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"
GRAFICOS_DIR = RESULTS_DIR / "graficos"

# ── Dataset ─────────────────────────────────────────────
DATA_FILE   = DATA_DIR / "diabetes.csv"
TARGET_COL  = "Outcome"
FEATURE_COLS = ["Glucose", "BMI", "Insulin", "Pregnancies", "Age"]
ZERO_COLS   = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

# ── Pré-processamento ────────────────────────────────────
IQR_FACTOR  = 1.5
TEST_SIZE   = 0.15
N_SPLITS    = 7

# ── Treinamento ──────────────────────────────────────────
RANDOM_STATE = 42
EPOCHS_LIST  = [200, 400, 800]
LEARNING_RATE = 0.01
BATCH_SIZE   = 32

# ── Artigo de referência ─────────────────────────────────
REFERENCE = {
    "titulo": "A comparison of machine learning algorithms for diabetes prediction",
    "autores": "Khanam, J.J. & Foo, S.Y.",
    "ano": 2021,
    "doi": "10.1016/j.icte.2021.02.004",
    "resultados": {
        "Decision Tree":        {"kfold": 0.7424, "split": 0.7314},
        "Random Forest":        {"kfold": 0.7496, "split": 0.7714},
        "Naive Bayes":          {"kfold": 0.7553, "split": 0.7828},
        "Logistic Regression":  {"kfold": 0.7682, "split": 0.7885},
        "KNN":                  {"kfold": 0.7510, "split": 0.7942},
        "AdaBoost":             {"kfold": 0.7396, "split": 0.7942},
        "SVM":                  {"kfold": 0.7682, "split": 0.7771},
        "Rede Neural (NN-2)":   {"kfold": 0.7600, "split": 0.8857},
    }
}
```

---

### `src/data_loader.py`
Responsável exclusivamente por carregar e validar o dataset.

```python
"""
data_loader.py — Carregamento e validação do dataset Pima Indians Diabetes.
"""

def load_dataset(filepath: Path) -> pd.DataFrame:
    """
    Carrega o dataset CSV e valida sua integridade.

    Parâmetros:
        filepath (Path): caminho para o arquivo diabetes.csv

    Retorna:
        pd.DataFrame: dataset carregado com 768 linhas × 9 colunas

    Lança:
        FileNotFoundError: se o arquivo não existir
        ValueError: se o número de colunas for diferente de 9
    """

def summarize_dataset(df: pd.DataFrame) -> None:
    """Exibe resumo estatístico, tipos e contagem de zeros por coluna."""
```

---

### `src/preprocessor.py`
Implementa o pipeline completo de pré-processamento. Cada etapa é uma função independente.

```python
"""
preprocessor.py — Pipeline de pré-processamento do dataset Pima Indians.
Segue metodologia do artigo Khanam & Foo (2021).
"""

def replace_zeros_with_nan(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Substitui zeros biologicamente impossíveis por NaN."""

def impute_with_median(df: pd.DataFrame) -> pd.DataFrame:
    """Preenche NaN com a mediana de cada coluna."""

def remove_outliers_iqr(df: pd.DataFrame, factor: float = 1.5) -> pd.DataFrame:
    """Remove outliers usando o método IQR."""

def select_features(df: pd.DataFrame, feature_cols: list, target_col: str):
    """Separa features (X) e alvo (y)."""

def normalize_features(X_train, X_test) -> tuple:
    """Aplica MinMaxScaler [0,1] no treino e transforma o teste."""

def split_dataset(X, y, test_size: float, random_state: int) -> tuple:
    """Divide em treino e teste com estratificação."""

def run_full_pipeline(df: pd.DataFrame) -> dict:
    """Executa o pipeline completo e retorna X_train, X_test, y_train, y_test e o scaler."""
```

---

### `src/models.py`
Define e instancia todos os modelos clássicos.

```python
"""
models.py — Definição dos modelos de Machine Learning clássicos.
"""

def get_classical_models() -> dict:
    """
    Retorna dicionário com todos os 7 modelos configurados.

    Retorna:
        dict: {nome_modelo: instância_sklearn}
    """

def train_and_evaluate_kfold(model, X, y, n_splits: int) -> dict:
    """Treina com K-fold e retorna métricas médias."""

def train_and_evaluate_split(model, X_train, X_test, y_train, y_test) -> dict:
    """Treina com split e retorna métricas."""

def run_all_models(X, y, X_train, X_test, y_train, y_test) -> pd.DataFrame:
    """Executa todos os 7 modelos e retorna DataFrame com resultados."""
```

---

### `src/neural_network.py`
Define e treina as 3 arquiteturas de Rede Neural.

```python
"""
neural_network.py — Implementação das Redes Neurais com Keras/TensorFlow.
Segue arquiteturas definidas em Khanam & Foo (2021).
"""

def build_nn_model(n_hidden_layers: int, input_dim: int = 5) -> Sequential:
    """
    Constrói modelo de Rede Neural conforme arquitetura do artigo.

    Parâmetros:
        n_hidden_layers (int): número de camadas ocultas (1, 2 ou 3)
        input_dim (int): número de features de entrada (padrão: 5)

    Retorna:
        Sequential: modelo Keras compilado
    """

def train_nn(model, X_train, y_train, epochs: int, batch_size: int) -> History:
    """Treina o modelo e retorna o histórico de treinamento."""

def evaluate_nn(model, X_test, y_test) -> dict:
    """Avalia o modelo no conjunto de teste e retorna métricas."""

def run_all_nn_experiments(X_train, X_test, y_train, y_test) -> pd.DataFrame:
    """Executa todos os experimentos de NN (3 arquiteturas × 3 configurações de épocas)."""
```

---

### `src/evaluator.py`
Calcula métricas e salva resultados.

```python
"""
evaluator.py — Cálculo de métricas e persistência de resultados.
"""

def compute_metrics(y_true, y_pred, y_proba=None) -> dict:
    """Calcula acurácia, precisão, recall, F1 e AUC-ROC."""

def save_results(results_df: pd.DataFrame, filepath: Path) -> None:
    """Salva ou atualiza o arquivo resultados.csv."""

def compare_with_reference(results_df: pd.DataFrame, reference: dict) -> pd.DataFrame:
    """Gera tabela comparativa entre resultados obtidos e do artigo."""
```

---

### `src/visualizer.py`
Gera e salva todos os gráficos.

```python
"""
visualizer.py — Geração de visualizações e gráficos do projeto.
"""

def plot_accuracy_comparison(results_df: pd.DataFrame, save_path: Path) -> None:
    """Gráfico de barras: acurácia K-fold vs Split por modelo."""

def plot_confusion_matrix(y_true, y_pred, model_name: str, save_path: Path) -> None:
    """Matriz de confusão do modelo especificado."""

def plot_roc_curve(y_true, y_proba, model_name: str, save_path: Path) -> None:
    """Curva ROC com valor de AUC."""

def plot_learning_curves(history_dict: dict, save_path: Path) -> None:
    """Curvas de loss e accuracy por época para as 3 arquiteturas de NN."""

def plot_feature_importance(model, feature_names: list, save_path: Path) -> None:
    """Importância das features para o Random Forest."""

def plot_article_comparison(results_df: pd.DataFrame, reference: dict, save_path: Path) -> None:
    """Tabela visual comparando resultados obtidos vs artigo."""
```

---

## 4. Sequência de Execução dos Notebooks

Os notebooks devem ser executados nesta ordem:

```
01_eda.ipynb
    └── Saída: prints de análise, gráficos de distribuição
        └── NÃO modifica dados

02_preprocessamento.ipynb
    └── Entrada: data/diabetes.csv
    └── Saída: X_train, X_test, y_train, y_test (em memória)
        └── Exporta: nada em disco (dados não versionados)

03_modelos_classicos.ipynb
    └── Entrada: pipeline de preprocessamento
    └── Saída: results/resultados.csv (modelos clássicos)
             results/graficos/01_*.png ... 02_*.png ... 06_*.png

04_rede_neural.ipynb
    └── Entrada: pipeline de preprocessamento
    └── Saída: results/resultados.csv (atualizado com NN)
             results/graficos/03_*.png, 04_*.png, 05_*.png, 07_*.png
```

---

## 5. Padrão de Importação nos Notebooks

Todo notebook deve usar este bloco no início para importar os módulos de `src/`:

```python
# ── Configuração de caminhos ─────────────────────────────
import sys
from pathlib import Path

# Detectar ambiente (Colab vs local)
import os
IS_COLAB = 'COLAB_GPU' in os.environ or 'COLAB_RELEASE_TAG' in os.environ

if IS_COLAB:
    # No Colab: fazer clone do repositório
    # !git clone https://github.com/SEU_USUARIO/diabetes-ml-prediction.git
    # %cd diabetes-ml-prediction
    ROOT = Path('/content/diabetes-ml-prediction')
else:
    ROOT = Path('..').resolve()

sys.path.append(str(ROOT))

# ── Imports do projeto ───────────────────────────────────
from src.config import *
from src.data_loader import load_dataset, summarize_dataset
from src.preprocessor import run_full_pipeline
```