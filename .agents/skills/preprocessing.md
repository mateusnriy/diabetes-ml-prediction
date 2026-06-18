# Skill: Pré-processamento de Dados

Esta skill auxilia o Antigravity CLI (agy) no pré-processamento do dataset Pima Indians Diabetes.

## Instruções para o agy

1. **Substituição de Zeros**: Substituir zeros por `NaN` nas colunas: `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin` e `BMI`.
2. **Imputação pela Mediana**: Preencher valores ausentes (`NaN`) com a mediana respectiva de cada coluna.
3. **Remoção de Outliers por IQR**:
   - Calcular limites superior/inferior baseados no fator `1.5` sobre o IQR.
   - Filtrar e remover linhas contendo outliers em qualquer atributo (redução esperada de 768 para 699 linhas).
4. **Seleção de Features**: Reter apenas as 5 features mais relevantes: `Glucose`, `BMI`, `Insulin`, `Pregnancies`, `Age`.
5. **Divisão Estratificada**: Dividir em treino (85%) e teste (15%) mantendo a proporção de classes de `Outcome`.
6. **Normalização**:
   - Ajustar o `MinMaxScaler` **apenas** no conjunto de treino.
   - Transformar conjuntos de treino e teste.
