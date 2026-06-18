"""
neural_network.py — Implementação de Redes Neurais com Keras/TensorFlow.

Este módulo implementa a definição, compilação, treinamento e avaliação de 3
arquiteturas de Redes Neurais (NN-1, NN-2 e NN-3) sob 3 configurações de épocas
(200, 400 e 800) conforme o artigo base.

Referência: Khanam & Foo (2021), DOI: 10.1016/j.icte.2021.02.004
"""

# ── Stdlib ───────────────────────────────────────────────
import random

# ── Terceiros ────────────────────────────────────────────
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tensorflow.keras.callbacks import History
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD

# ── Locais ───────────────────────────────────────────────
from src.config import BATCH_SIZE, EPOCHS_LIST, LEARNING_RATE, RANDOM_STATE

# Configura as sementes globais de aleatoriedade para reprodutibilidade
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)


def build_nn_model(n_hidden_layers: int, input_dim: int = 5) -> Sequential:
    """
    Constrói e compila o modelo de Rede Neural com base no número de camadas ocultas.

    Parâmetros:
        n_hidden_layers (int): número de camadas ocultas (1, 2 ou 3)
        input_dim (int): número de features na entrada (padrão: 5)

    Retorna:
        Sequential: modelo Keras compilado com SGD (learning_rate=0.01)
    """
    if n_hidden_layers == 1:
        model = Sequential(
            [
                Dense(
                    5,
                    activation="relu",
                    input_shape=(input_dim,),
                    name="entrada",
                ),
                Dense(5, activation="relu", name="oculta_1"),
                Dense(1, activation="sigmoid", name="saida"),
            ],
            name="NN-1",
        )
    elif n_hidden_layers == 2:
        model = Sequential(
            [
                Dense(
                    5,
                    activation="relu",
                    input_shape=(input_dim,),
                    name="entrada",
                ),
                Dense(26, activation="relu", name="oculta_1"),
                Dense(5, activation="relu", name="oculta_2"),
                Dense(1, activation="sigmoid", name="saida"),
            ],
            name="NN-2",
        )
    elif n_hidden_layers == 3:
        model = Sequential(
            [
                Dense(
                    5,
                    activation="relu",
                    input_shape=(input_dim,),
                    name="entrada",
                ),
                Dense(16, activation="relu", name="oculta_1"),
                Dense(10, activation="relu", name="oculta_2"),
                Dense(5, activation="relu", name="oculta_3"),
                Dense(1, activation="sigmoid", name="saida"),
            ],
            name="NN-3",
        )
    else:
        raise ValueError("O número de camadas ocultas deve ser 1, 2 ou 3.")

    model.compile(
        optimizer=SGD(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_nn1(input_dim: int = 5) -> Sequential:
    """
    Constrói modelo NN-1 (1 camada oculta).
    """
    return build_nn_model(1, input_dim)


def build_nn2(input_dim: int = 5) -> Sequential:
    """
    Constrói modelo NN-2 (2 camadas ocultas).
    """
    return build_nn_model(2, input_dim)


def build_nn3(input_dim: int = 5) -> Sequential:
    """
    Constrói modelo NN-3 (3 camadas ocultas).
    """
    return build_nn_model(3, input_dim)


def train_nn(
    model: Sequential,
    X_train: np.ndarray,
    y_train: pd.Series,
    epochs: int,
    batch_size: int,
) -> History:
    """
    Treina o modelo de Rede Neural fixando as sementes antes da execução.

    Parâmetros:
        model (Sequential): modelo Keras compilado
        X_train (np.ndarray): features de treino normalizadas
        y_train (pd.Series): alvo de treino
        epochs (int): número de épocas
        batch_size (int): tamanho do lote

    Retorna:
        History: histórico de treinamento
    """
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    tf.random.set_seed(RANDOM_STATE)

    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.15,
        verbose=0,
    )
    return history


def evaluate_nn(
    model: Sequential, X_test: np.ndarray, y_test: pd.Series
) -> dict[str, float]:
    """
    Avalia a Rede Neural no conjunto de teste e calcula as métricas correspondentes.

    Parâmetros:
        model (Sequential): modelo treinado
        X_test (np.ndarray): features de teste normalizadas
        y_test (pd.Series): rótulo alvo de teste

    Retorna:
        dict[str, float]: dicionário contendo acurácia, precisão, recall, f1 e auc-roc
    """
    y_proba = model.predict(X_test, verbose=0).flatten()
    y_pred = (y_proba > 0.5).astype(int)

    return {
        "acuracia": float(accuracy_score(y_test, y_pred)),
        "precisao": float(precision_score(y_test, y_pred, average="weighted")),
        "recall": float(recall_score(y_test, y_pred, average="weighted")),
        "f1": float(f1_score(y_test, y_pred, average="weighted")),
        "auc": float(roc_auc_score(y_test, y_proba)),
    }


def run_all_nn_experiments(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: pd.Series,
    y_test: pd.Series,
) -> tuple[pd.DataFrame, dict[str, dict]]:
    """
    Executa os experimentos de Rede Neural de todas as 9 combinações.

    Parâmetros:
        X_train (np.ndarray): features de treino normalizadas
        X_test (np.ndarray): features de teste normalizadas
        y_train (pd.Series): alvo de treino
        y_test (pd.Series): alvo de teste

    Retorna:
        tuple[pd.DataFrame, dict[str, dict]]:
            - DataFrame com os resultados agregados das redes neurais
            - Dicionário mapeando 'NN-X (Y épocas)' para o dicionário history.history correspondente
    """
    architectures = [("NN-1", build_nn1), ("NN-2", build_nn2), ("NN-3", build_nn3)]
    results = []
    history_dict = {}

    for nn_name, build_fn in architectures:
        for epochs in EPOCHS_LIST:
            exp_name = f"{nn_name} ({epochs} épocas)"
            print(f"→ Treinando {exp_name}...")

            # Cria modelo limpo para evitar vazamento de pesos
            model = build_fn(input_dim=X_train.shape[1])

            history = train_nn(
                model, X_train, y_train, epochs=epochs, batch_size=BATCH_SIZE
            )
            history_dict[exp_name] = history.history

            metrics = evaluate_nn(model, X_test, y_test)
            metrics["modelo"] = exp_name
            metrics["metodo"] = "split"

            results.append(metrics)
            print(f"  - Acurácia de Teste: {metrics['acuracia']:.4f}")

    df_results = pd.DataFrame(results)
    cols = ["modelo", "metodo", "acuracia", "precisao", "recall", "f1", "auc"]
    df_results = df_results[cols]

    return df_results, history_dict
