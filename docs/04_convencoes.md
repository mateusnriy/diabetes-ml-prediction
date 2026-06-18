# 04 — Convenções de Desenvolvimento

> O Antigravity CLI deve seguir todas as convenções abaixo ao gerar qualquer código, commit ou documentação.

---

## 1. Convenções de Código Python

### 1.1 Nomenclatura

| Elemento              | Convenção         | Exemplo                        |
|-----------------------|-------------------|--------------------------------|
| Variáveis             | `snake_case`      | `train_accuracy`               |
| Funções               | `snake_case`      | `remove_outliers_iqr()`        |
| Classes               | `PascalCase`      | `DiabetesPreprocessor`         |
| Constantes            | `UPPER_SNAKE`     | `RANDOM_STATE = 42`            |
| Módulos/arquivos      | `snake_case`      | `data_loader.py`               |
| Notebooks             | `NN_nome.ipynb`   | `03_modelos_classicos.ipynb`   |
| Features ML           | `X`, `X_train`    | Convenção padrão de ML         |
| Target ML             | `y`, `y_train`    | Convenção padrão de ML         |

**Proibido:**
- Variáveis de 1 letra (exceto `X`, `y`, `i`, `j` em loops simples)
- Nomes em português para variáveis e funções
- Abreviações obscuras (`df2`, `tmp`, `aux`)

**Permitido:**
- `df` para DataFrames principais
- `ax` para eixos matplotlib
- `fig` para figuras matplotlib

---

### 1.2 Docstrings (obrigatório em todo módulo e função)

**Padrão para módulos:**

```python
"""
nome_modulo.py — Descrição em uma linha do que o módulo faz.

Descrição mais detalhada se necessário. Mencione a qual etapa
do pipeline este módulo pertence.

Referência: Khanam & Foo (2021), DOI: 10.1016/j.icte.2021.02.004
"""
```

**Padrão para funções:**

```python
def remove_outliers_iqr(df: pd.DataFrame, factor: float = 1.5) -> pd.DataFrame:
    """
    Remove outliers usando o método Interquartile Range (IQR).

    Parâmetros:
        df (pd.DataFrame): dataset após imputação de valores ausentes
        factor (float): multiplicador do IQR para definir limites (padrão: 1.5)

    Retorna:
        pd.DataFrame: dataset sem outliers

    Exemplo:
        >>> df_clean = remove_outliers_iqr(df, factor=1.5)
        >>> print(df_clean.shape)  # (699, 9) aproximadamente
    """
```

---

### 1.3 Comentários em Notebooks

Células de código em notebooks devem ter comentário de seção no topo:

```python
# ── 1. Carregamento do dataset ───────────────────────────
import pandas as pd
from pathlib import Path
from src.data_loader import load_dataset

df = load_dataset(DATA_FILE)
print(f"Dataset carregado: {df.shape[0]} linhas × {df.shape[1]} colunas")
```

Regra: cada bloco lógico separado por `# ── Título ──` com linha de traços.

---

### 1.4 Formatação

- Indentação: **4 espaços** (nunca tabs)
- Comprimento máximo de linha: **88 caracteres** (padrão Black)
- Linha em branco: **2 linhas** entre funções de nível superior; **1 linha** entre métodos de classe
- Imports: ordenados por (1) stdlib, (2) terceiros, (3) locais, separados por linha em branco

```python
# ── Stdlib ───────────────────────────────────────────────
import os
import sys
from pathlib import Path

# ── Terceiros ────────────────────────────────────────────
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ── Locais ───────────────────────────────────────────────
from src.config import RANDOM_STATE, FEATURE_COLS
from src.preprocessor import run_full_pipeline
```

---

### 1.5 Type Hints (obrigatório em funções de `src/`)

```python
# Correto
def split_dataset(
    X: np.ndarray,
    y: pd.Series,
    test_size: float = 0.15,
    random_state: int = 42
) -> tuple[np.ndarray, np.ndarray, pd.Series, pd.Series]:
    ...

# Incorreto
def split_dataset(X, y, test_size, random_state):
    ...
```

---

## 2. Convenções de Células de Notebook

### Estrutura padrão de cada notebook

```
Célula 1: Markdown — Título e descrição do notebook
Célula 2: Código   — Instalação de dependências (pip install)
Célula 3: Código   — Imports e configuração de caminhos
Célula 4: Código   — Configurações globais (seeds, estilos)
Célula 5+: Conteúdo do notebook
Última célula: Markdown — Resumo dos resultados e próximos passos
```

### Célula de título (obrigatória — primeira célula de todo notebook)

```markdown
# 03 — Modelos Clássicos de Machine Learning

**Projeto:** Predição de Diabetes com ML  
**Artigo base:** Khanam & Foo (2021) — DOI: 10.1016/j.icte.2021.02.004  
**Objetivo:** Implementar e comparar 7 algoritmos de ML clássicos usando K-fold
cross-validation (k=7) e train/test split (85/15), reproduzindo os experimentos
do artigo de referência.

**Modelos:** Decision Tree, KNN, Random Forest, Naive Bayes, AdaBoost,
Logistic Regression, SVM
```

### Célula de instalação (obrigatória — segunda célula de todo notebook)

```python
# Instalar dependências (necessário no Google Colab)
import subprocess
subprocess.run(["pip", "install", "-r", "../requirements.txt", "-q"], check=True)
print("✓ Dependências instaladas")
```

### Célula de configuração de reprodutibilidade

```python
# ── Reprodutibilidade ────────────────────────────────────
import random
import numpy as np
import tensorflow as tf

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
print(f"✓ Seeds fixadas: {SEED}")
```

---

## 3. Convenções de Commits Git

### Formato

```
<tipo>(<escopo>): <descrição curta em português>

[corpo opcional em português]
[rodapé opcional]
```

### Tipos permitidos

| Tipo       | Quando usar                                        |
|------------|----------------------------------------------------|
| `feat`     | nova funcionalidade ou notebook                    |
| `fix`      | correção de bug                                    |
| `docs`     | alteração em documentação                          |
| `refactor` | refatoração sem mudança de comportamento           |
| `chore`    | tarefas de manutenção (requirements, gitignore)    |
| `results`  | adição ou atualização de resultados e gráficos     |
| `report`   | alterações no relatório técnico                    |

### Escopos do projeto

`eda`, `prep`, `models`, `nn`, `eval`, `viz`, `docs`, `config`, `report`

### Exemplos de commits válidos

```bash
git commit -m "feat(prep): implementa remoção de outliers por IQR"
git commit -m "feat(models): adiciona 7 modelos clássicos com avaliação K-fold"
git commit -m "feat(nn): implementa NN-2 com 400 épocas — melhor resultado"
git commit -m "results(eval): adiciona resultados comparativos com artigo original"
git commit -m "docs(readme): atualiza tabela de resultados obtidos"
git commit -m "fix(prep): corrige imputação de mediana por coluna (era global)"
git commit -m "report: adiciona seção de análise de resultados"
```

### Commits proibidos

```bash
# Proibido — muito vago
git commit -m "update"
git commit -m "fix bug"
git commit -m "wip"
git commit -m "changes"

# Proibido — dados não devem ser versionados
git add data/diabetes.csv
```

---

## 4. Convenções de Variáveis de ML

Para manter consistência com o ecossistema scikit-learn e Keras:

| Variável       | Descrição                                      |
|----------------|------------------------------------------------|
| `X`            | Matriz de features completa (após normalização)|
| `y`            | Vetor de labels completo                       |
| `X_train`      | Features de treino                             |
| `X_test`       | Features de teste                              |
| `y_train`      | Labels de treino                               |
| `y_test`       | Labels de teste                                |
| `y_pred`       | Predições do modelo                            |
| `y_proba`      | Probabilidades preditas (para AUC-ROC)         |
| `model`        | Instância de um modelo                         |
| `history`      | Histórico de treinamento Keras                 |
| `results_df`   | DataFrame com métricas de todos os modelos     |
| `scaler`       | Instância do MinMaxScaler                      |
| `kfold`        | Instância do StratifiedKFold                   |

---

## 5. Convenções de Gráficos

Todo gráfico gerado deve seguir este padrão:

```python
# Estilo padrão do projeto
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = ['#1A56A0', '#2E7D32', '#E65100', '#6A1B9A', '#00838F', '#AD1457', '#37474F']

# Tamanho padrão
fig, ax = plt.subplots(figsize=(12, 6))

# Títulos
ax.set_title('Título do Gráfico', fontsize=14, fontweight='bold', pad=16)
ax.set_xlabel('Eixo X', fontsize=11)
ax.set_ylabel('Eixo Y', fontsize=11)

# Salvar sempre antes de exibir
plt.tight_layout()
plt.savefig(GRAFICOS_DIR / 'nome_arquivo.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"✓ Gráfico salvo em: {GRAFICOS_DIR / 'nome_arquivo.png'}")
```

---

## 6. Convenções de Saída no Terminal

Todo output relevante deve seguir o padrão visual com prefixo de status:

```python
print(f"✓ Dataset carregado: {df.shape[0]} registros")
print(f"✓ Pré-processamento concluído: {X_train.shape[0]} treino | {X_test.shape[0]} teste")
print(f"✓ Modelo treinado: {model_name} — Acurácia: {accuracy:.4f}")
print(f"⚠ Aviso: {n_outliers} outliers removidos")
print(f"✗ Erro: arquivo não encontrado em {filepath}")
```

Prefixos:
- `✓` — sucesso / conclusão
- `⚠` — aviso / atenção
- `✗` — erro
- `→` — informação intermediária / progresso