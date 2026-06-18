# 05 — Especificação do Dataset

> O Antigravity CLI deve consultar este arquivo ao gerar notebooks de EDA e pré-processamento.

---

## 1. Identificação

| Campo             | Valor                                                                    |
|-------------------|--------------------------------------------------------------------------|
| Nome              | Pima Indians Diabetes Database                                            |
| Fonte original    | National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK) |
| Repositório       | UCI Machine Learning Repository                                           |
| Kaggle            | https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database      |
| Licença           | CC0 — Domínio Público                                                     |
| Formato           | CSV, sem cabeçalho próprio (usar nomes abaixo)                            |
| Tamanho           | 768 registros × 9 colunas                                                 |
| População         | Mulheres de herança indígena Pima, com 21 anos ou mais                   |

---

## 2. Descrição dos Atributos

| #  | Nome                      | Tipo     | Faixa         | Média  | Zeros suspeitos |
|----|---------------------------|----------|---------------|--------|-----------------|
| 1  | `Pregnancies`             | Inteiro  | 0 – 17        | 3,85   | Não (0 é válido)|
| 2  | `Glucose`                 | Real     | 0 – 199       | 120,89 | **Sim (5)**     |
| 3  | `BloodPressure`           | Real     | 0 – 122       | 69,11  | **Sim (35)**    |
| 4  | `SkinThickness`           | Real     | 0 – 99        | 20,54  | **Sim (227)**   |
| 5  | `Insulin`                 | Real     | 0 – 846       | 79,80  | **Sim (374)**   |
| 6  | `BMI`                     | Real     | 0 – 67,1      | 32,00  | **Sim (11)**    |
| 7  | `DiabetesPedigreeFunction`| Real     | 0,078 – 2,42  | 0,47   | Não             |
| 8  | `Age`                     | Inteiro  | 21 – 81       | 33     | Não             |
| 9  | `Outcome`                 | Binário  | 0 ou 1        | —      | **Variável alvo**|

**Legenda:** Zeros suspeitos = valores biologicamente impossíveis que representam dados ausentes.

---

## 3. Distribuição das Classes

| Classe | Significado     | Contagem | Percentual |
|--------|-----------------|----------|------------|
| 0      | Sem diabetes    | 500      | 65,1%      |
| 1      | Com diabetes    | 268      | 34,9%      |
| **Total** | —            | **768**  | 100%       |

⚠️ **Dataset desbalanceado** — proporção de aproximadamente 1,87:1 (negativo:positivo).

---

## 4. Features Selecionadas pelo Artigo

O artigo Khanam & Foo (2021) aplicou correlação de Pearson e selecionou as **5 features mais relevantes**:

| Feature         | Correlação com Outcome | Mantida? |
|-----------------|------------------------|----------|
| `Glucose`       | 0,484                  | ✅ Sim   |
| `BMI`           | 0,316                  | ✅ Sim   |
| `Insulin`       | 0,261                  | ✅ Sim   |
| `Pregnancies`   | 0,226                  | ✅ Sim   |
| `Age`           | 0,224                  | ✅ Sim   |
| `SkinThickness` | 0,193                  | ❌ Não   |
| `BloodPressure` | 0,183                  | ❌ Não   |
| `DiabetesPedigreeFunction` | 0,178      | ❌ Não   |

**Cutoff de seleção:** 0,20 (features abaixo deste valor foram removidas).

---

## 5. Pré-processamento Detalhado

### 5.1 Zeros biologicamente impossíveis

Os seguintes atributos não podem ter valor zero em um exame clínico real:
- `Glucose` — glicose zero indica morte
- `BloodPressure` — pressão zero indica morte
- `SkinThickness` — espessura de pele zero é impossível
- `Insulin` — valor zero pode ser ausente
- `BMI` — IMC zero é impossível

**Ação:** Substituir zeros por `NaN` e imputar com a **mediana** de cada coluna.

```python
ZERO_COLS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

df[ZERO_COLS] = df[ZERO_COLS].replace(0, np.nan)
df.fillna(df.median(), inplace=True)
```

---

### 5.2 Remoção de Outliers (IQR)

O artigo original removeu 45 outliers e 26 valores extremos, resultando em **699 instâncias**.

```python
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1

mask = ~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)
df_clean = df[mask]

print(f"→ Registros antes: 768")
print(f"→ Registros removidos: {768 - len(df_clean)}")
print(f"→ Registros após remoção: {len(df_clean)}")  # esperado: ~699
```

---

### 5.3 Normalização

Aplicar MinMaxScaler para escalar todas as features para o intervalo [0, 1]:

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)  # fit APENAS no treino
X_test_scaled  = scaler.transform(X_test)        # transform no teste
```

⚠️ **Regra crítica:** O `fit_transform` deve ser aplicado **somente no conjunto de treino**.
O conjunto de teste usa apenas `transform` para evitar data leakage.

---

### 5.4 Divisão Treino/Teste

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.15,        # 15% para teste
    stratify=y,            # mantém proporção das classes
    random_state=42
)

print(f"✓ Treino: {X_train.shape[0]} registros ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"✓ Teste:  {X_test.shape[0]} registros ({X_test.shape[0]/len(X)*100:.1f}%)")
```

---

## 6. Estatísticas Esperadas Após Pré-processamento

| Atributo      | Média (antes) | Média (depois normaliz.) | Desvio Padrão (depois) |
|---------------|---------------|--------------------------|------------------------|
| Pregnancies   | 0,23          | 0,23                     | 0,20                   |
| Glucose       | 0,48          | 0,48                     | 0,19                   |
| Insulin       | 0,50          | 0,50                     | 0,18                   |
| BMI           | 0,35          | 0,35                     | 0,17                   |
| Age           | 0,20          | 0,20                     | 0,19                   |

Fonte: Khanam & Foo (2021), Tabela 4.

---

## 7. Análise Exploratória Esperada (EDA)

O notebook `01_eda.ipynb` deve responder às seguintes perguntas:

1. Qual a distribuição de cada atributo? (histogramas)
2. Existe correlação forte entre atributos? (heatmap)
3. As classes estão balanceadas? (countplot)
4. Quais atributos diferem entre diabéticos e não-diabéticos? (boxplots)
5. Quais colunas têm zeros suspeitos e quantos? (tabela)
6. Após tratamento, a distribuição mudou? (antes/depois)

---

## 8. Como Baixar o Dataset no Google Colab

```python
# Opção 1: Upload manual
from google.colab import files
uploaded = files.upload()  # selecionar diabetes.csv

# Opção 2: Via Kaggle API (requer configuração prévia)
# !pip install kaggle -q
# from google.colab import files
# files.upload()  # fazer upload de kaggle.json
# !mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
# !kaggle datasets download -d uciml/pima-indians-diabetes-database
# !unzip pima-indians-diabetes-database.zip

# Opção 3: URL direta (se disponível)
import urllib.request
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
# Nota: este arquivo não tem cabeçalho — adicionar manualmente
cols = ["Pregnancies","Glucose","BloodPressure","SkinThickness",
        "Insulin","BMI","DiabetesPedigreeFunction","Age","Outcome"]
df = pd.read_csv(url, header=None, names=cols)
```