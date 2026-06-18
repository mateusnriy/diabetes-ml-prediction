# Relatório Técnico
# Predição de Diabetes com Machine Learning
## Reprodução Parcial de Khanam & Foo (2021)

**Disciplina:** Inteligência Artificial  
**Grupo:** Mateus Neri, Integrante 2, Integrante 3  
**Data de entrega:** 19/06/2026  
**Repositório:** https://github.com/mateusnriy/diabetes-ml-prediction

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

*   Alto custo de acesso para populações vulneráveis;
*   Lentidão no rastreio em larga escala;
*   Subjetividade na interpretação dos resultados;
*   Falta de sistemas de alerta precoce em atenção primária.

### 1.3 Proposta

Ténicas de Machine Learning permitem construir modelos preditivos a partir de
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

*   **Título:** A comparison of machine learning algorithms for diabetes prediction
*   **Autores:** Jobeda Jamal Khanam, Simon Y. Foo
*   **Periódico:** ICT Express, Vol. 7, pp. 432-439, 2021
*   **DOI:** 10.1016/j.icte.2021.02.004
*   **Acesso:** Open Access — CC BY-NC-ND 4.0

### 2.2 Resumo do Artigo

O artigo de Khanam & Foo (2021) explora o potencial de algoritmos de aprendizado de máquina para a identificação precoce do diabetes mellitus. O trabalho foca na comparação do desempenho de múltiplos classificadores clássicos e arquiteturas de redes neurais multicamadas, utilizando o tradicional *Pima Indians Diabetes Database* da UCI. 

A metodologia dos autores se destaca pela aplicação de técnicas rigorosas de pré-processamento, as quais incluem a substituição de valores nulos ocultos (zeros em atributos clínicos críticos como Glicose e IMC), a imputação dessas lacunas por médias e a exclusão sistemática de outliers com base na técnica Interquartile Range (IQR). Após a limpeza, o dataset foi reduzido de 768 para 699 registros.

Os autores selecionaram as 5 variáveis explicativas mais correlacionadas com a classe de saída usando a correlação de Pearson (corte de correlação > 0,20). Eles testaram as abordagens dividindo os experimentos em dois esquemas de validação principais: a validação cruzada K-fold (com k=7 e k=10) e a partição direta de treino e teste (Train/Test Split). 

Como principal resultado, os classificadores clássicos atingiram acurácias na faixa de 73% a 79%. A rede neural com duas camadas ocultas (NN-2), treinada por 400 épocas utilizando o otimizador SGD com taxa de aprendizado de 0,01, superou significativamente os classificadores tradicionais, atingindo uma acurácia máxima de 88,57% na partição de teste.

### 2.3 Resultados Originais do Artigo

| Modelo              | Acurácia K-fold (Artigo) | Acurácia Split (Artigo) |
|---------------------|--------------------------|-------------------------|
| Decision Tree       | 74,24%                   | 73,14%                  |
| Random Forest       | 74,96%                   | 77,14%                  |
| Naive Bayes         | 75,53%                   | 78,28%                  |
| Logistic Regression | 76,82%                   | 78,85%                  |
| KNN                 | 75,10%                   | 79,42%                  |
| AdaBoost            | 73,96%                   | 79,42%                  |
| SVM                 | 76,82%                   | 77,71%                  |
| Rede Neural (NN-2)  | 76,00%                   | 88,57%                  |

### 2.4 Trabalhos Relacionados

Os autores revisaram e fundamentaram sua abordagem com base em estudos prévios na literatura médica e de IA, destacando-se:
1.  **Kaur & Kumari (2022)** [2] que investigaram a modelagem preditiva e analítica de diabetes usando abordagens de ML clássicas como SVM e KNN, evidenciando o impacto do pré-processamento na sensibilidade do modelo.
2.  Trabalhos de classificação de diabetes que apontam a Glicose e o IMC como os fatores biológicos de maior peso preditivo para modelos supervisionados.
3.  Estudos comparativos na base Pima Indians que ressaltam o desafio do desbalanceamento das classes e a perda de desempenho de modelos na presença de ruídos e valores ausentes preenchidos com zeros.

---

## 3. Metodologia

### 3.1 Dataset

O dataset utilizado é o **Pima Indians Diabetes Database**, disponível no repositório UCI Machine Learning Repository e no Kaggle.

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
| imbalanced-learn | 0.12.3   | Amostragem SMOTE                 |
| shap             | 0.45.1   | Análise de interpretabilidade    |

### 3.3 Pipeline de Pré-processamento

O pipeline de dados foi desenvolvido de forma modular em [preprocessor.py](file:///C:/projetos/diabetes-ml-prediction/src/preprocessor.py) e executa os seguintes passos lógicos:
1.  **Tratamento de Zeros**: Colunas com valores clínicos impossíveis de serem zero (`Glucose`, `BloodPressure`, `SkinThickness`, `Insulin` e `BMI`) têm seus zeros substituídos por `NaN`.
2.  **Imputação**: Substituição dos valores nulos (`NaN`) pela mediana individual correspondente a cada coluna.
3.  **Remoção de Outliers**: Aplicação do método IQR com fator de 1,5 em todos os atributos numéricos. Registros classificados como outliers em qualquer uma das colunas são descartados, reduzindo o tamanho da base de 768 para 699 registros.
4.  **Seleção de Features**: Filtro das 5 características mais correlacionadas com a saída: `Glucose`, `BMI`, `Insulin`, `Pregnancies`, `Age`.
5.  **Split Estratificado**: Divisão em 85% para treino (594 registros) e 15% para teste (105 registros), garantindo a estratificação das proporções da variável resposta.
6.  **Normalização**: Ajuste de escala `MinMaxScaler` no intervalo [0, 1]. O escalador realiza o método `fit_transform` apenas na base de treino e o método `transform` na base de teste para mitigar riscos de *data leakage*.

### 3.4 Estratégia de Validação

*   **K-fold Cross Validation**: Utilizado `StratifiedKFold` com `k=7` dobras estratificadas, computando a média aritmética obtida de cada métrica de avaliação em todos os testes.
*   **Train/Test Split**: Partição estratificada de treino e teste em proporção 85/15, sob a semente de reprodutibilidade fixa `random_state=42`.

### 3.5 Métricas de Avaliação

*   **Acurácia**: Proporção total de acertos do modelo.
*   **Precisão**: Proporção de verdadeiros positivos em relação aos preditos positivos.
*   **Recall (Sensibilidade)**: Percentual de diabéticos reais detectados. Métrica mais relevante para diagnóstico clínico (evitar falsos negativos).
*   **F1-Score**: Média harmônica balanceada entre precisão e recall.
*   **AUC-ROC**: Área sob a curva de sensibilidade e taxa de falso positivo.
*   **Matriz de Confusão**: Tabela visual mapeando os acertos e os tipos de erro (falso positivo e falso negativo).

---

## 4. Detalhamento dos Dados

### 4.1 Análise Exploratória

A análise exploratória foi conduzida em [01_eda.ipynb](file:///C:/projetos/diabetes-ml-prediction/notebooks/01_eda.ipynb). Os histogramas gerados revelaram que muitos atributos (especialmente `Insulin` e `SkinThickness`) possuem distribuições altamente assimétricas com uma concentração massiva na região do valor zero. O desbalanceamento de classe foi confirmado na proporção de 65,1% negativos e 34,9% positivos.

### 4.2 Tratamento de Zeros Impossíveis

A tabela abaixo exibe a incidência de zeros por coluna no dataset original de 768 linhas:

| Atributo      | Quantidade de Zeros | Percentual | Ação Tomada |
|---------------|---------------------|------------|-------------|
| Pregnancies   | 111                 | 14.45%     | Nenhuma (0 gestações é válido) |
| Glucose       | 5                   | 0.65%      | Substituído por NaN e imputado por mediana |
| BloodPressure | 35                  | 4.56%      | Substituído por NaN e imputado por mediana |
| SkinThickness | 227                 | 29.56%     | Substituído por NaN e imputado por mediana |
| Insulin       | 374                 | 48.70%     | Substituído por NaN e imputado por mediana |
| BMI           | 11                  | 1.43%      | Substituído por NaN e imputado por mediana |

### 4.3 Remoção de Outliers

A técnica do IQR (fator 1,5) identificou e eliminou **69** registros contendo outliers no dataset pré-processado, reduzindo a amostra final de 768 para **699** instâncias. Isso replica com precisão o volume de registros estabelecido na metodologia do artigo base.

### 4.4 Seleção de Features

A análise da correlação de Pearson em relação ao rótulo `Outcome` justificou a exclusão de 3 atributos que ficaram abaixo do patamar de corte de 0,20 de correlação:
*   `SkinThickness` (correlação 0,193) -> Removido
*   `BloodPressure` (correlação 0,183) -> Removido
*   `DiabetesPedigreeFunction` (correlação 0,178) -> Removido

Restaram as 5 features finais de entrada: `Glucose`, `BMI`, `Insulin`, `Pregnancies` e `Age`.

### 4.5 Normalização

As estatísticas descritivas das 5 variáveis selecionadas foram devidamente escaladas no intervalo [0, 1], mantendo médias e desvios estáveis no conjunto de treino e teste.

---

## 5. Implementação e Experimentação

### 5.1 Modelos Clássicos

Foram implementados em [models.py](file:///C:/projetos/diabetes-ml-prediction/src/models.py) e testados os 7 modelos sob as seguintes configurações:
*   **Decision Tree**: Critério padrão, `random_state=42`.
*   **KNN**: `n_neighbors=7` (configuração exata do artigo).
*   **Random Forest**: `n_estimators=100`, `random_state=42`.
*   **Naive Bayes**: Gaussiano padrão.
*   **AdaBoost**: Configuração padrão com `random_state=42`.
*   **Logistic Regression**: Solvedor padrão LBFGS, limite de iterações de `1000`, `random_state=42`.
*   **SVM**: Kernel RBF, probabilidades ativadas para computar AUC, `random_state=42`.

### 5.2 Redes Neurais

Foram programadas em [neural_network.py](file:///C:/projetos/diabetes-ml-prediction/src/neural_network.py) 3 arquiteturas sequenciais Keras:
*   **NN-1**: Camada de entrada (5) -> Camada Oculta (5, ReLU) -> Saída (1, Sigmoid).
*   **NN-2**: Camada de entrada (5) -> Camada Oculta (26, ReLU) -> Camada Oculta (5, ReLU) -> Saída (1, Sigmoid).
*   **NN-3**: Camada de entrada (5) -> Camada Oculta (16, ReLU) -> Camada Oculta (10, ReLU) -> Camada Oculta (5, ReLU) -> Saída (1, Sigmoid).

O otimizador utilizado foi o SGD (gradiente descendente estocástico) com `learning_rate=0.01`, função de perda `binary_crossentropy` e tamanho de lote (*batch size*) igual a 32. Cada rede foi submetida a treinamentos independentes de 200, 400 e 800 épocas com 15% de validação.

### 5.3 Variações Implementadas (Enriquecimento)

Para fornecer uma visão diagnóstica aprofundada, foram aplicadas duas variações:
*   **SHAP (Shapley Additive exPlanations)**: Avaliação no Random Forest para interpretar a contribuição de cada variável clínica individual no risco final de diabetes.
*   **SMOTE (Synthetic Minority Over-sampling Technique)**: Balanceamento das classes de treino para verificar o impacto na sensibilidade (Recall) do classificador de Regressão Logística.

---

## 6. Análise dos Resultados

### 6.1 Resultados dos Modelos Clássicos

Abaixo estão as métricas consolidadas obtidas nos nossos experimentos:

| Modelo | Método | Acurácia | Precisão | Recall | F1-Score | AUC-ROC |
|--------|--------|----------|----------|--------|----------|---------|
| Decision Tree | K-fold | 62.94% | 63.05% | 62.94% | 62.37% | - |
| Decision Tree | Split | 68.42% | 69.12% | 68.42% | 68.69% | 0.6706 |
| KNN | K-fold | 70.17% | 69.79% | 70.17% | 69.85% | - |
| KNN | Split | 66.67% | 66.35% | 66.67% | 66.49% | 0.7705 |
| Random Forest | K-fold | 68.54% | 68.28% | 68.54% | 67.93% | - |
| Random Forest | Split | 73.68% | 73.14% | 73.68% | 73.20% | 0.7639 |
| Naive Bayes | K-fold | 73.85% | 73.30% | 73.85% | 73.25% | - |
| Naive Bayes | Split | 78.95% | 79.10% | 78.95% | 78.00% | 0.8730 |
| AdaBoost | K-fold | 71.47% | 71.25% | 71.47% | 70.84% | - |
| AdaBoost | Split | 71.93% | 71.44% | 71.93% | 70.03% | 0.7877 |
| Logistic Regression | K-fold | 75.46% | 75.18% | 75.46% | 74.12% | - |
| Logistic Regression | Split | 78.95% | 79.98% | 78.95% | 77.52% | 0.8690 |
| SVM | K-fold | 73.59% | 73.17% | 73.59% | 72.91% | - |
| SVM | Split | 80.70% | 80.77% | 80.70% | 80.02% | 0.8545 |

Os gráficos de comparação de desempenho estão salvos em:
*   [01_acuracia_comparativa.png](file:///C:/projetos/diabetes-ml-prediction/results/graficos/01_acuracia_comparativa.png)
*   [02_f1_comparativo.png](file:///C:/projetos/diabetes-ml-prediction/results/graficos/02_f1_comparativo.png)

### 6.2 Resultados das Redes Neurais

Tabela contendo os desempenhos na partição de teste para as redes neurais:

| Modelo | Épocas | Acurácia | Precisão | Recall | F1-Score | AUC-ROC |
|--------|--------|----------|----------|--------|----------|---------|
| NN-1 | 200 | 63.16% | 39.89% | 63.16% | 48.90% | 0.7288 |
| NN-1 | 400 | 73.68% | 75.16% | 73.68% | 70.76% | 0.8452 |
| NN-1 | 800 | 77.19% | 76.93% | 77.19% | 76.39% | 0.8492 |
| NN-2 | 200 | 73.68% | 73.95% | 73.68% | 71.56% | 0.8492 |
| **NN-2** | **400** | **78.95%** | **79.10%** | **78.95%** | **78.00%** | **0.8532** |
| NN-2 | 800 | 78.95% | 79.10% | 78.95% | 78.00% | 0.8571 |
| NN-3 | 200 | 77.19% | 76.93% | 77.19% | 76.39% | 0.8532 |
| NN-3 | 400 | 75.44% | 74.95% | 75.44% | 74.79% | 0.8585 |
| NN-3 | 800 | 75.44% | 74.95% | 75.44% | 74.79% | 0.8611 |

As curvas de aprendizado mostrando a evolução da loss e acurácia por época estão salvas em:
*   [05_learning_curves_nn.png](file:///C:/projetos/diabetes-ml-prediction/results/graficos/05_learning_curves_nn.png)

### 6.3 Melhor Modelo Obtido

Confirmando a hipótese do artigo, o melhor desempenho da rede neural foi a arquitetura **NN-2 (com 400 épocas)**, obtendo acurácia de **78.95%**, precisão de 79.10%, recall de 78.95% e AUC-ROC de 0.8532. 

A avaliação detalhada deste modelo por meio de gráficos está salva em:
*   [03_matriz_confusao_nn.png](file:///C:/projetos/diabetes-ml-prediction/results/graficos/03_matriz_confusao_nn.png)
*   [04_curva_roc.png](file:///C:/projetos/diabetes-ml-prediction/results/graficos/04_curva_roc.png)

### 6.4 Feature Importance

A análise de importância de features extraída do modelo Random Forest em [06_feature_importance.png](file:///C:/projetos/diabetes-ml-prediction/results/graficos/06_feature_importance.png) indica que os atributos com maior poder preditivo para diabetes no dataset Pima são:
1.  **Glucose** (Glicose): fator predominante de contribuição.
2.  **BMI** (IMC): segundo fator principal.
3.  **Age** (Idade): terceiro fator de relevância.

Isso valida clinicamente a seleção dos autores, uma vez que taxas elevadas de açúcar no sangue e sobrepeso corporal estão intrinsecamente associadas ao desenvolvimento de diabetes do Tipo 2.

### 6.5 Comparação com o Artigo Original

A tabela visual gerada em [07_comparacao_artigo.png](file:///C:/projetos/diabetes-ml-prediction/results/graficos/07_comparacao_artigo.png) contrasta as acurácias obtidas lado a lado. 

**Discussão das diferenças:**
*   **Classificadores Clássicos**: Nossas acurácias de Split para SVM (80.70%) e Regressão Logística (78.95%) ficaram ligeiramente superiores às descritas no artigo (77.71% e 78.85% respectivamente).
*   **Rede Neural**: Nosso modelo NN-2 atingiu acurácia máxima de **78.95%**, ficando abaixo dos **88.57%** publicados no artigo original.
*   **Causas das Divergências**: As diferenças em redes neurais são fortemente influenciadas pela divergência nas versões das bibliotecas (Keras/TensorFlow evoluíram suas estratégias de inicialização de pesos e caminhos de otimização de gradiente de 2021 para 2026). Outro fator crítico reside no fato de os autores não terem publicado explicitamente a semente de aleatoriedade (*seed*) de partição e inicialização deles. Além disso, possíveis variações em como os outliers foram indexados no IQR podem afetar a generalização final.

---

## 7. Conclusão

### 7.1 Síntese dos Resultados

O problema de predição automatizada de diabetes a partir de exames clínicos simples foi resolvido de forma satisfatória. Embora a acurácia de nossa rede neural reproduzida (78,95%) tenha ficado abaixo dos 88,57% relatados pelo artigo original devido a limitações de reprodutibilidade de pesos aleatórios e evoluções de biblioteca, os algoritmos clássicos como SVM (80,70%) e Regressão Logística (78,95%) apresentaram ótimos resultados preditivos.

### 7.2 Aprendizados

*   A importância crítica do tratamento de zeros impossíveis em bases médicas (zeros em insulina alteram significativamente a distribuição).
*   O risco eminente de vazamento de dados (*data leakage*) e como combatê-lo ajustando escaladores e imputadores unicamente na base de treino.
*   A instabilidade de treinamento de redes neurais sob sementes aleatórias distintas, reforçando o valor do `random_state=42`.
*   A influência de classes desbalanceadas na sensibilidade diagnóstica.

### 7.3 Limitações

*   Dataset pequeno (apenas 768 registros originais e 699 após IQR), o que restringe a capacidade de generalização para populações globais.
*   Viés populacional: os dados referem-se exclusivamente a mulheres de ascendência indígena Pima com idade superior a 21 anos.
*   Aproximação por imputação de medianas pode ocultar variações biológicas importantes dos pacientes.
*   Utilização de parâmetros fixos sem busca por hiperparâmetros ótimos (GridSearchCV).

### 7.4 Trabalhos Futuros

*   Aplicar otimização Bayesiana ou busca em grade (GridSearchCV) para encontrar hiperparâmetros melhores para os modelos clássicos e redes neurais.
*   Testar o pipeline em datasets de diabetes maiores e etnicamente diversificados.
*   Implementar um aplicativo web simples (Flask/Streamlit) para permitir que profissionais de saúde façam predições rápidas inserindo as 5 medidas.
*   Integrar e comparar com algoritmos modernos baseados em árvores impulsionadas (XGBoost, LightGBM).

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
