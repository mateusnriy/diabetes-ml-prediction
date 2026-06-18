# 06 — Especificação dos Modelos de Machine Learning

> O Antigravity CLI deve consultar este arquivo ao gerar `notebooks/03_modelos_classicos.ipynb`,
> `notebooks/04_rede_neural.ipynb` e `src/models.py`, `src/neural_network.py`.

---

## 1. Modelos Clássicos de ML

### 1.1 Configuração dos Modelos

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from src.config import RANDOM_STATE

CLASSICAL_MODELS = {
    "Decision Tree": DecisionTreeClassifier(
        random_state=RANDOM_STATE
    ),
    "KNN": KNeighborsClassifier(
        n_neighbors=7       # valor k=7 usado no artigo original
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE
    ),
    "Naive Bayes": GaussianNB(),
    "AdaBoost": AdaBoostClassifier(
        random_state=RANDOM_STATE
    ),
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE
    ),
    "SVM": SVC(
        kernel='rbf',
        probability=True,   # necessário para calcular AUC-ROC
        random_state=RANDOM_STATE
    ),
}
```

---

### 1.2 Método de Avaliação K-fold

```python
from sklearn.model_selection import StratifiedKFold, cross_validate
from src.config import N_SPLITS, RANDOM_STATE

kfold = StratifiedKFold(
    n_splits=N_SPLITS,    # k=7 conforme artigo
    shuffle=True,
    random_state=RANDOM_STATE
)

SCORING_METRICS = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']

for model_name, model in CLASSICAL_MODELS.items():
    cv_results = cross_validate(
        model, X, y,
        cv=kfold,
        scoring=SCORING_METRICS,
        return_train_score=False
    )
    # Extrair médias de cada métrica
```

---

### 1.3 Método de Avaliação Train/Test Split

```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report
)

for model_name, model in CLASSICAL_MODELS.items():
    model.fit(X_train, y_train)
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "modelo":    model_name,
        "metodo":    "split",
        "acuracia":  accuracy_score(y_test, y_pred),
        "precisao":  precision_score(y_test, y_pred, average='weighted'),
        "recall":    recall_score(y_test, y_pred, average='weighted'),
        "f1":        f1_score(y_test, y_pred, average='weighted'),
        "auc":       roc_auc_score(y_test, y_proba),
    }
```

---

### 1.4 Resultados Esperados do Artigo (referência de comparação)

| Modelo              | Precisão K-fold | Recall K-fold | F1 K-fold | Acurácia K-fold | Acurácia Split |
|---------------------|-----------------|---------------|-----------|-----------------|----------------|
| Decision Tree       | 0,739           | 0,742         | 0,741     | 74,24%          | 73,14%         |
| Random Forest       | 0,744           | 0,750         | 0,746     | 74,96%          | 77,14%         |
| Naive Bayes         | 0,753           | 0,755         | 0,754     | 75,53%          | 78,28%         |
| Logistic Regression | 0,761           | 0,768         | 0,761     | 76,82%          | 78,85%         |
| KNN                 | 0,747           | 0,751         | 0,749     | 75,10%          | 79,42%         |
| AdaBoost            | 0,730           | 0,740         | 0,730     | 73,96%          | 79,42%         |
| SVM                 | 0,761           | 0,768         | 0,759     | 76,82%          | 77,71%         |

---

## 2. Redes Neurais (Keras / TensorFlow)

### 2.1 Arquiteturas

Implementar as 3 arquiteturas abaixo, cada uma com 3 configurações de épocas (200, 400, 800):

#### NN-1 — Uma Camada Oculta

```
Entrada (5) → Oculta (5, ReLU) → Saída (1, Sigmoid)
```

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD

def build_nn1(input_dim: int = 5) -> Sequential:
    model = Sequential([
        Dense(5, activation='relu', input_shape=(input_dim,), name='entrada'),
        Dense(5, activation='relu', name='oculta_1'),
        Dense(1, activation='sigmoid', name='saida')
    ], name='NN-1')
    model.compile(
        optimizer=SGD(learning_rate=0.01),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model
```

---

#### NN-2 — Duas Camadas Ocultas (melhor resultado esperado)

```
Entrada (5) → Oculta (26, ReLU) → Oculta (5, ReLU) → Saída (1, Sigmoid)
```

```python
def build_nn2(input_dim: int = 5) -> Sequential:
    model = Sequential([
        Dense(5, activation='relu', input_shape=(input_dim,), name='entrada'),
        Dense(26, activation='relu', name='oculta_1'),
        Dense(5, activation='relu', name='oculta_2'),
        Dense(1, activation='sigmoid', name='saida')
    ], name='NN-2')
    model.compile(
        optimizer=SGD(learning_rate=0.01),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model
```

---

#### NN-3 — Três Camadas Ocultas

```
Entrada (5) → Oculta (16, ReLU) → Oculta (10, ReLU) → Oculta (5, ReLU) → Saída (1, Sigmoid)
```

```python
def build_nn3(input_dim: int = 5) -> Sequential:
    model = Sequential([
        Dense(5,  activation='relu', input_shape=(input_dim,), name='entrada'),
        Dense(16, activation='relu', name='oculta_1'),
        Dense(10, activation='relu', name='oculta_2'),
        Dense(5,  activation='relu', name='oculta_3'),
        Dense(1,  activation='sigmoid', name='saida')
    ], name='NN-3')
    model.compile(
        optimizer=SGD(learning_rate=0.01),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model
```

---

### 2.2 Treinamento

```python
EPOCHS_LIST = [200, 400, 800]
BATCH_SIZE  = 32

all_nn_results = []

for nn_name, build_fn in [("NN-1", build_nn1), ("NN-2", build_nn2), ("NN-3", build_nn3)]:
    for epochs in EPOCHS_LIST:
        # Recriar modelo a cada experimento para evitar acumulação de pesos
        model = build_fn(input_dim=X_train.shape[1])

        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=BATCH_SIZE,
            validation_split=0.15,
            verbose=0  # silencioso para loop
        )

        y_pred  = (model.predict(X_test) > 0.5).astype(int).flatten()
        y_proba = model.predict(X_test).flatten()

        result = {
            "modelo":   f"{nn_name} ({epochs} épocas)",
            "metodo":   "split",
            "acuracia": accuracy_score(y_test, y_pred),
            "precisao": precision_score(y_test, y_pred, average='weighted'),
            "recall":   recall_score(y_test, y_pred, average='weighted'),
            "f1":       f1_score(y_test, y_pred, average='weighted'),
            "auc":      roc_auc_score(y_test, y_proba),
        }
        all_nn_results.append(result)
        print(f"✓ {nn_name} | {epochs} épocas → Acurácia: {result['acuracia']:.4f}")
```

---

### 2.3 Resultados Esperados das Redes Neurais

| Versão | Épocas | Acurácia Esperada | Acurácia Treino | Acurácia Teste |
|--------|--------|-------------------|-----------------|----------------|
| NN-1   | 200    | 0,838             | 76,43%          | 83,81%         |
| NN-1   | 400    | 0,848             | 77,27%          | 84,76%         |
| NN-1   | 800    | 0,829             | 79,46%          | 82,86%         |
| NN-2   | 200    | 0,876             | 76,77%          | 87,62%         |
| **NN-2** | **400** | **0,886**      | **78,96%**      | **88,57%**     |
| NN-2   | 800    | 0,857             | 81,65%          | 87,62%         |
| NN-3   | 200    | 0,829             | 76,77%          | 82,86%         |
| NN-3   | 400    | 0,838             | 83,00%          | 83,81%         |
| NN-3   | 800    | 0,790             | 87,04%          | 79,05%         |

Fonte: Khanam & Foo (2021), Tabela 8.

**Melhor modelo:** NN-2 com 400 épocas → **88,57% de acurácia** (learning_rate=0,01).

---

## 3. Variações Adicionais (para enriquecer o trabalho)

Implemente pelo menos **uma** das variações abaixo para demonstrar análise crítica:

### Variação A — Comparação com SMOTE

```python
from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state=RANDOM_STATE)
X_resampled, y_resampled = sm.fit_resample(X_train, y_train)

print(f"→ Treino original: {y_train.value_counts().to_dict()}")
print(f"→ Treino com SMOTE: {pd.Series(y_resampled).value_counts().to_dict()}")
```

### Variação B — XGBoost (não está no artigo, mas serve como comparação)

```python
from xgboost import XGBClassifier

xgb = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=RANDOM_STATE
)
```

### Variação C — Feature Importance com SHAP

```python
import shap

rf_model = CLASSICAL_MODELS["Random Forest"]
rf_model.fit(X_train, y_train)

explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values[1], X_test, feature_names=FEATURE_COLS)
```