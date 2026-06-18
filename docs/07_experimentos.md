# 07 — Plano de Experimentos e Métricas

> O Antigravity CLI deve consultar este arquivo ao gerar células de avaliação nos notebooks.

---

## 1. Experimentos Planejados

### EXP-01 — Modelos Clássicos com K-fold (k=7)

| Campo           | Valor                                              |
|-----------------|----------------------------------------------------|
| Objetivo        | Reproduzir Tabela 6 (K-fold) do artigo             |
| Modelos         | DT, KNN, RF, NB, AdaBoost, LR, SVM                |
| Validação       | StratifiedKFold(n_splits=7)                        |
| Métricas        | Acurácia, Precisão, Recall, F1-Score               |
| Saída           | `results/resultados.csv` — linhas com metodo=kfold |

---

### EXP-02 — Modelos Clássicos com Train/Test Split (85/15)

| Campo           | Valor                                              |
|-----------------|----------------------------------------------------|
| Objetivo        | Reproduzir Tabela 6 (Split) do artigo              |
| Modelos         | DT, KNN, RF, NB, AdaBoost, LR, SVM                |
| Validação       | train_test_split(test_size=0.15, stratify=y)       |
| Métricas        | Acurácia, Precisão, Recall, F1-Score, AUC-ROC      |
| Saída           | `results/resultados.csv` — linhas com metodo=split |

---

### EXP-03 — Redes Neurais (3 arquiteturas × 3 épocas)

| Campo           | Valor                                              |
|-----------------|----------------------------------------------------|
| Objetivo        | Reproduzir Tabela 8 do artigo                      |
| Arquiteturas    | NN-1, NN-2, NN-3                                   |
| Épocas          | 200, 400, 800                                      |
| Learning rate   | 0,01 (fixo, conforme artigo)                       |
| Otimizador      | SGD                                                |
| Validação       | validation_split=0.15 no treinamento               |
| Saída           | `results/resultados.csv` + curvas de aprendizado   |

---

### EXP-04 — Comparação Final com o Artigo

| Campo           | Valor                                              |
|-----------------|----------------------------------------------------|
| Objetivo        | Gerar tabela comparativa lado a lado               |
| Fonte A         | Resultados obtidos (results/resultados.csv)        |
| Fonte B         | config.REFERENCE["resultados"]                     |
| Saída           | `results/graficos/07_comparacao_artigo.png`        |

---

### EXP-05 — Variação: Impacto do SMOTE

| Campo           | Valor                                              |
|-----------------|----------------------------------------------------|
| Objetivo        | Verificar se balanceamento melhora recall          |
| Método          | SMOTE no conjunto de treino                        |
| Modelos         | RF e LR (representativos)                          |
| Comparação      | Com SMOTE vs sem SMOTE                             |
| Métrica foco    | Recall (mais importante em contexto médico)        |

---

## 2. Matriz Completa de Experimentos

```
                    K-FOLD       SPLIT       SMOTE
Decision Tree         ✓           ✓            -
KNN                   ✓           ✓            -
Random Forest         ✓           ✓            ✓
Naive Bayes           ✓           ✓            -
AdaBoost              ✓           ✓            -
Logistic Regression   ✓           ✓            ✓
SVM                   ✓           ✓            -
NN-1 (200 épocas)     -           ✓            -
NN-1 (400 épocas)     -           ✓            -
NN-1 (800 épocas)     -           ✓            -
NN-2 (200 épocas)     -           ✓            -
NN-2 (400 épocas)     -           ✓            -  ← MELHOR MODELO
NN-2 (800 épocas)     -           ✓            -
NN-3 (200 épocas)     -           ✓            -
NN-3 (400 épocas)     -           ✓            -
NN-3 (800 épocas)     -           ✓            -
```

---

## 3. Definição das Métricas

### Acurácia

```
Acurácia = (TP + TN) / (TP + TN + FP + FN)
```
Percentual de predições corretas sobre o total.

---

### Precisão (Precision)

```
Precisão = TP / (TP + FP)
```
Dos pacientes classificados como diabéticos, quantos realmente são.

---

### Recall (Sensibilidade)

```
Recall = TP / (TP + FN)
```
Dos pacientes diabéticos reais, quantos o modelo identificou.
⭐ **Métrica mais importante no contexto médico** — minimizar falsos negativos.

---

### F1-Score

```
F1 = 2 × (Precisão × Recall) / (Precisão + Recall)
```
Média harmônica entre precisão e recall. Ideal para datasets desbalanceados.

---

### AUC-ROC

Área sob a curva ROC. Valores:
- 1,0 = classificador perfeito
- 0,9+ = excelente
- 0,8–0,9 = muito bom
- 0,7–0,8 = bom
- 0,5 = classificador aleatório

---

### Matriz de Confusão

```
                  Predito Negativo   Predito Positivo
Real Negativo          TN                  FP
Real Positivo          FN                  TP
```

| Sigla | Significado         | Impacto no contexto                     |
|-------|---------------------|-----------------------------------------|
| TP    | Verdadeiro Positivo | Diabético corretamente identificado     |
| TN    | Verdadeiro Negativo | Não diabético corretamente identificado |
| FP    | Falso Positivo      | Não diabético classificado como diabético (alarme falso) |
| FN    | Falso Negativo      | Diabético classificado como saudável (perigoso!) |

---

## 4. Estrutura do Arquivo `results/resultados.csv`

```csv
modelo,metodo,acuracia,precisao,recall,f1,auc
Decision Tree,kfold,0.7424,0.739,0.742,0.741,
Decision Tree,split,0.7314,0.735,0.731,0.733,0.712
...
NN-2 (400 épocas),split,0.8857,0.881,0.886,0.883,0.921
```

**Regras:**
- AUC fica vazio para K-fold (não calculado por padrão no cross_validate)
- Uma linha por modelo por método de avaliação
- Valores com 4 casas decimais
- Arquivo é atualizado (não sobrescrito) ao rodar cada notebook

---

## 5. Código de Persistência dos Resultados

```python
import pandas as pd
from pathlib import Path
from src.config import RESULTS_DIR

def save_results(new_results: list[dict], filepath: Path = RESULTS_DIR / "resultados.csv") -> None:
    """
    Salva resultados no CSV, adicionando ao arquivo existente se já houver dados.

    Parâmetros:
        new_results (list[dict]): lista de dicionários com métricas
        filepath (Path): caminho do arquivo CSV de saída
    """
    df_new = pd.DataFrame(new_results)

    if filepath.exists():
        df_existing = pd.read_csv(filepath)
        # Remover linhas existentes com mesmo modelo+método (para atualizar)
        mask = ~(
            df_existing['modelo'].isin(df_new['modelo']) &
            df_existing['metodo'].isin(df_new['metodo'])
        )
        df_final = pd.concat([df_existing[mask], df_new], ignore_index=True)
    else:
        df_final = df_new

    filepath.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(filepath, index=False, float_format='%.4f')
    print(f"✓ Resultados salvos em: {filepath} ({len(df_final)} registros)")
```

---

## 6. Checklist de Conclusão dos Experimentos

Marque cada item ao concluir:

- [ ] EXP-01: K-fold para todos os 7 modelos clássicos
- [ ] EXP-02: Split 85/15 para todos os 7 modelos clássicos
- [ ] EXP-03: Todas as 9 combinações de NN (3 arq. × 3 épocas)
- [ ] EXP-04: Tabela comparativa com artigo gerada e salva
- [ ] EXP-05: Pelo menos uma variação (SMOTE, SHAP ou XGBoost)
- [ ] Todos os 7 gráficos salvos em `results/graficos/`
- [ ] `results/resultados.csv` completo e atualizado
- [ ] Métricas discutidas no relatório técnico