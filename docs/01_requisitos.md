# 01 — Requisitos do Projeto

> Referência: Antigravity CLI deve consultar este arquivo ao gerar qualquer módulo de código ou notebook.

---

## 1. Requisitos Funcionais (RF)

Descrevem o que o sistema deve fazer.

---

### RF-01 — Carregamento de Dados

- O sistema deve carregar o arquivo `diabetes.csv` a partir do caminho `data/diabetes.csv`.
- Deve exibir shape, tipos de colunas e primeiras linhas ao carregar.
- Deve validar se o arquivo existe antes de prosseguir, lançando erro descritivo caso contrário.

```python
# Comportamento esperado
df = load_dataset("data/diabetes.csv")
# → DataFrame com 768 linhas × 9 colunas
# → Colunas: Pregnancies, Glucose, BloodPressure, SkinThickness,
#            Insulin, BMI, DiabetesPedigreeFunction, Age, Outcome
```

---

### RF-02 — Análise Exploratória de Dados (EDA)

- O sistema deve gerar as seguintes visualizações obrigatórias no notebook `01_eda.ipynb`:
  - [ ] Distribuição de cada atributo (histograma)
  - [ ] Mapa de correlação (heatmap Pearson)
  - [ ] Contagem de classes (Outcome 0 vs 1)
  - [ ] Boxplots por classe para cada feature
  - [ ] Identificação e contagem de zeros suspeitos por coluna

---

### RF-03 — Pré-processamento

O pré-processamento deve seguir exatamente esta sequência:

| Passo | Operação                        | Colunas afetadas                                              | Método           |
|-------|---------------------------------|---------------------------------------------------------------|------------------|
| 1     | Substituir zeros por NaN        | Glucose, BloodPressure, SkinThickness, Insulin, BMI           | `.replace(0, NaN)` |
| 2     | Imputar valores ausentes        | Mesmas colunas                                                | Mediana por coluna |
| 3     | Remover outliers                | Todas as colunas numéricas                                    | IQR (1.5×)       |
| 4     | Selecionar features             | Glucose, BMI, Insulin, Pregnancies, Age                       | Correlação Pearson |
| 5     | Normalizar                      | Todas as features selecionadas                                | MinMaxScaler [0,1] |
| 6     | Dividir em treino/teste         | —                                                             | StratifiedKFold  |

- Após remoção de outliers, o dataset deve ter aproximadamente 699 instâncias.
- A divisão deve ser 85% treino / 15% teste.
- Usar `random_state=42` em todas as operações com aleatoriedade.

---

### RF-04 — Implementação dos Modelos Clássicos

O sistema deve implementar e avaliar os seguintes 7 algoritmos:

| ID    | Modelo               | Classe scikit-learn                          |
|-------|----------------------|----------------------------------------------|
| M-01  | Decision Tree        | `DecisionTreeClassifier()`                   |
| M-02  | K-Nearest Neighbors  | `KNeighborsClassifier(n_neighbors=7)`        |
| M-03  | Random Forest        | `RandomForestClassifier(n_estimators=100)`   |
| M-04  | Naive Bayes          | `GaussianNB()`                               |
| M-05  | AdaBoost             | `AdaBoostClassifier()`                       |
| M-06  | Logistic Regression  | `LogisticRegression(max_iter=1000)`          |
| M-07  | SVM                  | `SVC(kernel='rbf', probability=True)`        |

Cada modelo deve ser avaliado com:
- K-fold Cross Validation (`k=7`, `StratifiedKFold`)
- Train/Test Split (85%/15%)

---

### RF-05 — Implementação da Rede Neural

O sistema deve implementar 3 arquiteturas de Rede Neural variando camadas ocultas:

| Versão | Camadas Ocultas | Neurônios por Camada | Ativação | Épocas testadas  |
|--------|-----------------|----------------------|----------|------------------|
| NN-1   | 1               | 5                    | ReLU     | 200, 400, 800    |
| NN-2   | 2               | 26, 5                | ReLU     | 200, 400, 800    |
| NN-3   | 3               | 16, 10, 5            | ReLU     | 200, 400, 800    |

Parâmetros fixos para todas as versões:
- Entrada: 5 neurônios (uma por feature selecionada)
- Saída: 1 neurônio com ativação `sigmoid`
- Otimizador: `SGD` com `learning_rate=0.01`
- Loss: `binary_crossentropy`
- Métrica: `accuracy`

---

### RF-06 — Avaliação e Métricas

Para cada modelo, o sistema deve calcular e registrar:

| Métrica          | Função scikit-learn                  |
|------------------|--------------------------------------|
| Acurácia         | `accuracy_score`                     |
| Precisão         | `precision_score(average='weighted')`|
| Recall           | `recall_score(average='weighted')`   |
| F1-Score         | `f1_score(average='weighted')`       |
| AUC-ROC          | `roc_auc_score`                      |
| Matriz Confusão  | `confusion_matrix`                   |

Todos os resultados devem ser salvos em `results/resultados.csv` com colunas:
`modelo, metodo, acuracia, precisao, recall, f1, auc`

---

### RF-07 — Visualizações de Resultados

O sistema deve gerar e salvar em `results/graficos/` os seguintes gráficos:

| Arquivo                        | Descrição                                             |
|--------------------------------|-------------------------------------------------------|
| `01_acuracia_comparativa.png`  | Barras: acurácia de todos os modelos (K-fold vs Split)|
| `02_f1_comparativo.png`        | Barras: F1-Score de todos os modelos                  |
| `03_matriz_confusao_nn.png`    | Matriz de confusão do melhor modelo (NN-2, 400 épocas)|
| `04_curva_roc.png`             | Curva ROC do melhor modelo com valor AUC              |
| `05_learning_curves_nn.png`    | Loss e accuracy por época para NN-1, NN-2, NN-3       |
| `06_feature_importance.png`    | Importância das features (Random Forest)              |
| `07_comparacao_artigo.png`     | Tabela visual: resultados obtidos vs artigo original  |

---

### RF-08 — Comparação com o Artigo Original

O sistema deve gerar uma tabela comparativa entre os resultados obtidos e os do artigo Khanam & Foo (2021):

| Modelo              | Acurácia Artigo (K-fold) | Acurácia Artigo (Split) |
|---------------------|--------------------------|-------------------------|
| Decision Tree       | 74,24%                   | 73,14%                  |
| Random Forest       | 74,96%                   | 77,14%                  |
| Naive Bayes         | 75,53%                   | 78,28%                  |
| Logistic Regression | 76,82%                   | 78,85%                  |
| KNN                 | 75,10%                   | 79,42%                  |
| AdaBoost            | 73,96%                   | 79,42%                  |
| SVM                 | 76,82%                   | 77,71%                  |
| Rede Neural (NN-2)  | 76,00% (k=10)            | 88,57%                  |

---

### RF-09 — Relatório Técnico

O sistema deve gerar o arquivo `reports/relatorio_tecnico.md` com as seções:

1. Introdução / Problematização
2. Revisão do Artigo Base
3. Metodologia
4. Detalhamento dos Dados
5. Implementação e Experimentação
6. Análise dos Resultados
7. Conclusão
8. Referências Bibliográficas

---

## 2. Requisitos Não Funcionais (RNF)

Descrevem como o sistema deve se comportar.

---

### RNF-01 — Reprodutibilidade

- Todo experimento deve ser reproduzível com `random_state=42`.
- O arquivo `requirements.txt` deve ter versões exatas (ex: `scikit-learn==1.4.2`).
- Cada notebook deve ser executável do início ao fim em ordem sequencial das células.
- O tempo de execução total de todos os notebooks não deve exceder 15 minutos em CPU padrão.

---

### RNF-02 — Organização do Código

- Funções reutilizáveis devem estar em `src/`, não duplicadas em notebooks.
- Notebooks devem importar de `src/` usando `sys.path.append('../')`.
- Nenhum notebook deve ter mais de 30 células.
- Células de código não devem ter mais de 30 linhas.

---

### RNF-03 — Qualidade do Código

- Todo módulo em `src/` deve ter docstring no topo.
- Toda função deve ter docstring com parâmetros e retorno.
- Não usar variáveis com nomes de 1 letra (exceto `X`, `y` por convenção de ML).
- Usar f-strings para formatação de strings (não `.format()` nem `%`).

---

### RNF-04 — Rastreabilidade

- Toda célula que gera um gráfico deve salvar o arquivo `.png` correspondente.
- Toda célula que treina um modelo deve exibir as métricas no output.
- O arquivo `results/resultados.csv` deve ser atualizado ao final de cada notebook de modelagem.

---

### RNF-05 — Compatibilidade de Ambiente

- O projeto deve funcionar no Google Colab sem configurações adicionais.
- Não usar caminhos absolutos; usar sempre `pathlib.Path` com caminhos relativos.
- O notebook deve detectar automaticamente se está no Colab ou local e ajustar caminhos.

```python
# Padrão de detecção de ambiente
import os
IS_COLAB = 'COLAB_GPU' in os.environ or 'COLAB_RELEASE_TAG' in os.environ
DATA_PATH = Path('/content/data') if IS_COLAB else Path('../data')
```

---

### RNF-06 — Documentação

- Todo arquivo em `src/` deve ter no mínimo 60% de cobertura de comentários.
- O `README.md` deve ter badges de: Python version, licença, status do projeto.
- Cada pasta deve ter seu próprio `README.md` explicando o conteúdo.

---

### RNF-07 — Segurança e Boas Práticas

- Nunca fazer commit de dados (`*.csv`, `*.xlsx`) — garantido pelo `.gitignore`.
- Nunca fazer commit de chaves de API ou credenciais.
- O arquivo `.gitignore` deve incluir: `data/*.csv`, `__pycache__/`, `.ipynb_checkpoints/`, `*.pyc`, `.env`.