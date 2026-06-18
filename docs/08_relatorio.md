# 08 — Estrutura do Relatório Técnico

> O Antigravity CLI deve usar este arquivo como template ao gerar `reports/relatorio_tecnico.md`.

---

## Template Completo do Relatório

```markdown
# Relatório Técnico
# Predição de Diabetes com Machine Learning
## Reprodução Parcial de Khanam & Foo (2021)

**Disciplina:** Análise e Aplicação de IA na Resolução de Problemas
**Grupo:** [Nome 1], [Nome 2], [Nome 3]
**Data de entrega:** 19/06/2026
**Repositório:** https://github.com/[usuario]/diabetes-ml-prediction

---

## 1. Introdução / Problematização

### 1.1 Contexto

O diabetes mellitus é uma das doenças crônicas mais prevalentes do mundo.
Segundo a Organização Mundial da Saúde (OMS), cerca de 1,6 milhão de pessoas
morrem anualmente em decorrência direta do diabetes [1]. A doença ocorre quando
o pâncreas não produz insulina suficiente (Tipo 1) ou quando o organismo não
consegue usar eficientemente a insulina produzida (Tipo 2).

Complicações de longo prazo incluem doenças cardiovasculares, insuficiência
renal, neuropatias e retinopatia. O diagnóstico precoce é essencial para
retardar essas complicações e melhorar a qualidade de vida dos pacientes.

### 1.2 Problema

O diagnóstico tradicional depende de exames laboratoriais e da avaliação
clínica por especialistas, o que representa:

- Alto custo de acesso para populações vulneráveis
- Lentidão no rastreio em larga escala
- Subjetividade na interpretação dos resultados
- Falta de sistemas de alerta precoce em atenção primária

### 1.3 Proposta

Técnicas de Machine Learning permitem construir modelos preditivos a partir de
dados clínicos simples (glicose, IMC, pressão arterial, idade, etc.), viabilizando
o rastreio automatizado de risco de diabetes com alta acurácia e baixo custo
computacional.

### 1.4 Objetivo do Trabalho

Reproduzir parcialmente os experimentos do artigo "A comparison of machine
learning algorithms for diabetes prediction" (Khanam & Foo, 2021), implementando
e comparando 7 algoritmos de ML clássicos e 3 arquiteturas de Redes Neurais
no dataset Pima Indians Diabetes.

---

## 2. Revisão do Artigo Base

### 2.1 Identificação

- **Título:** A comparison of machine learning algorithms for diabetes prediction
- **Autores:** Jobeda Jamal Khanam, Simon Y. Foo
- **Periódico:** ICT Express, Vol. 7, pp. 432-439, 2021
- **DOI:** 10.1016/j.icte.2021.02.004
- **Acesso:** Open Access — CC BY-NC-ND 4.0

### 2.2 Resumo do Artigo

[Descrever em 3-4 parágrafos: objetivo, metodologia, principais resultados e contribuições do artigo]

### 2.3 Resultados Originais

[Inserir tabela de resultados da seção 2.3 do documento docs/01_requisitos.md]

### 2.4 Trabalhos Relacionados

[Citar pelo menos 3 outros trabalhos do artigo que usaram abordagens similares]

---

## 3. Metodologia

### 3.1 Dataset

O dataset utilizado é o **Pima Indians Diabetes Database**, disponível no
repositório UCI Machine Learning Repository e no Kaggle.

| Propriedade    | Valor                                                    |
|----------------|----------------------------------------------------------|
| Origem         | National Institute of Diabetes (NIDDK), USA              |
| Registros      | 768                                                      |
| Atributos      | 9 (8 features + 1 target binário)                        |
| População      | Mulheres de herança indígena Pima, ≥ 21 anos             |
| Classes        | 0 = sem diabetes (500), 1 = com diabetes (268)           |

### 3.2 Ferramentas e Ambiente

| Ferramenta       | Versão   | Uso                              |
|------------------|----------|----------------------------------|
| Python           | 3.10+    | Linguagem principal              |
| pandas           | 2.2.2    | Manipulação de dados             |
| scikit-learn     | 1.4.2    | Algoritmos de ML e avaliação     |
| TensorFlow/Keras | 2.16.1   | Redes Neurais                    |
| matplotlib       | 3.8.4    | Visualizações                    |
| seaborn          | 0.13.2   | Visualizações estatísticas       |
| Google Colab     | —        | Ambiente de execução             |

### 3.3 Pipeline de Pré-processamento

[Descrever as 6 etapas da seção 5 de docs/05_dados.md, com tabela e justificativas]

### 3.4 Estratégia de Validação

**K-fold Cross Validation:**
- k = 7 folds
- Estratificado (mantém proporção das classes em cada fold)
- Calcula média das métricas em todos os folds

**Train/Test Split:**
- 85% treino, 15% teste
- Estratificado (stratify=y)
- random_state=42

### 3.5 Métricas de Avaliação

[Descrever as métricas da seção 3 de docs/07_experimentos.md]

---

## 4. Detalhamento dos Dados

### 4.1 Análise Exploratória

[Inserir histogramas de distribuição de cada atributo]

[Inserir heatmap de correlação]

[Inserir gráfico de distribuição das classes]

### 4.2 Tratamento de Zeros Impossíveis

[Tabela com quantidade de zeros por coluna e ação tomada]

### 4.3 Remoção de Outliers

[Descrever o método IQR, quantidade de outliers removidos e impacto no dataset]

### 4.4 Seleção de Features

[Tabela de correlação Pearson e justificativa para remoção das 3 features]

### 4.5 Normalização

[Descrever MinMaxScaler e estatísticas após normalização]

---

## 5. Implementação e Experimentação

### 5.1 Modelos Clássicos

[Para cada modelo, descrever: configuração, hiperparâmetros utilizados]

### 5.2 Redes Neurais

[Descrever as 3 arquiteturas, parâmetros de treinamento, e variações testadas]

### 5.3 Variações Implementadas

[Descrever qual variação foi implementada (SMOTE / SHAP / XGBoost) e por quê]

---

## 6. Análise dos Resultados

### 6.1 Resultados dos Modelos Clássicos

[Tabela comparativa: K-fold e Split — nossos resultados vs artigo]

[Inserir gráfico 01_acuracia_comparativa.png]

[Inserir gráfico 02_f1_comparativo.png]

### 6.2 Resultados das Redes Neurais

[Tabela com acurácia de treino e teste para todas as 9 combinações]

[Inserir gráfico 05_learning_curves_nn.png]

### 6.3 Melhor Modelo

[Análise detalhada do melhor modelo (esperado: NN-2 com 400 épocas)]

[Inserir gráfico 03_matriz_confusao_nn.png]

[Inserir gráfico 04_curva_roc.png]

### 6.4 Feature Importance

[Inserir gráfico 06_feature_importance.png]

[Análise: quais features foram mais relevantes e por quê]

### 6.5 Comparação com o Artigo Original

[Inserir gráfico 07_comparacao_artigo.png]

**Discussão das diferenças:**
- Diferenças encontradas entre nossos resultados e os do artigo
- Possíveis causas: versão das bibliotecas, implementação do IQR, seed
- O que os resultados sugerem sobre a reprodutibilidade do artigo

---

## 7. Conclusão

### 7.1 Síntese dos Resultados

[Responder: o problema de predição de diabetes foi resolvido satisfatoriamente?]

[Qual modelo apresentou melhor desempenho e por quê?]

[Os resultados foram reprodutíveis em relação ao artigo original?]

### 7.2 Aprendizados

[3-5 pontos sobre o que o grupo aprendeu com o projeto]

### 7.3 Limitações

- Dataset pequeno (768 registros) — pode não generalizar bem para outras populações
- Apenas mulheres de herança Pima — baixa diversidade demográfica
- Zeros como proxies de ausência reduzem qualidade dos dados
- Não testamos hiperparâmetros otimizados (ex: GridSearchCV)

### 7.4 Trabalhos Futuros

- Aplicar otimização de hiperparâmetros (GridSearchCV, Optuna)
- Testar com datasets maiores e mais diversificados
- Implementar interface web para predição em tempo real
- Explorar modelos mais recentes (Gradient Boosting, Transformers tabulares)

---

## 8. Referências Bibliográficas

[1] KHANAM, Jobeda Jamal; FOO, Simon Y. A comparison of machine learning
    algorithms for diabetes prediction. **ICT Express**, v. 7, n. 3, p. 432-439,
    2021. DOI: 10.1016/j.icte.2021.02.004.

[2] KAUR, Harleen; KUMARI, Vinita. Predictive modelling and analytics for
    diabetes using a machine learning approach. **Applied Computing and
    Informatics**, v. 18, n. 1/2, p. 90-100, 2022.
    DOI: 10.1016/j.aci.2018.12.004.

[3] WORLD HEALTH ORGANIZATION. Diabetes. Disponível em:
    https://www.who.int/health-topics/diabetes. Acesso em: 17 jun. 2026.

[4] LICHMAN, M. Pima Indians Diabetes Database. **UCI Machine Learning
    Repository**, 2013. Disponível em:
    https://archive.ics.uci.edu/dataset/34/diabetes.

[5] PEDREGOSA, F. et al. Scikit-learn: Machine Learning in Python.
    **Journal of Machine Learning Research**, v. 12, p. 2825-2830, 2011.
```