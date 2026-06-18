# Skill: Modelagem Clássica e Deep Learning

Esta skill auxilia o Antigravity CLI (agy) na fase de modelagem e experimentação do projeto.

## Instruções para o agy

1. **Definição de Modelos Clássicos**:
   - Configurar os 7 classificadores clássicos: Decision Tree, KNN (k=7), Random Forest (100 estimadores), Naive Bayes (Gaussiano), AdaBoost, Logistic Regression (max_iter=1000) e SVM (kernel RBF com probabilidades ativadas).
   - Usar `random_state=42` para todos os algoritmos com aleatoriedade.

2. **Validação e Avaliação**:
   - Executar validação K-fold com `k=7` estratificado.
   - Executar divisão treino/teste estratificada 85/15.
   - Calcular acurácia, precisão, recall, F1-score e AUC-ROC (apenas para split).

3. **Configuração de Redes Neurais**:
   - Implementar 3 arquiteturas no Keras (NN-1, NN-2 e NN-3) com ativação ReLU nas camadas ocultas e Sigmoid na saída.
   - Treinar usando SGD com taxa de aprendizado de `0.01` e batch size de `32`.
   - Testar sob 200, 400 e 800 épocas de treinamento.
   - Registrar métricas de validação usando `validation_split=0.15`.
