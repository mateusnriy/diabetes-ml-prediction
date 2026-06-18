"""
models.py — Definição e avaliação de modelos clássicos de Machine Learning.

Este módulo implementa a definição, treinamento e avaliação de 7 modelos clássicos
de Machine Learning (Decision Tree, KNN, Random Forest, Naive Bayes, AdaBoost,
Logistic Regression, SVM) através de validação K-fold e divisão treino/teste.

Referência: Khanam & Foo (2021), DOI: 10.1016/j.icte.2021.02.004
"""

# ── Terceiros ────────────────────────────────────────────
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# ── Locais ───────────────────────────────────────────────
from src.config import N_SPLITS, RANDOM_STATE


def get_classical_models() -> dict[str, any]:
    """
    Retorna um dicionário contendo as instâncias configuradas dos 7 modelos clássicos.

    Retorna:
        dict[str, any]: Dicionário no formato {nome_modelo: classificador_sklearn}
    """
    return {
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "KNN": KNeighborsClassifier(n_neighbors=7),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=RANDOM_STATE
        ),
        "Naive Bayes": GaussianNB(),
        "AdaBoost": AdaBoostClassifier(random_state=RANDOM_STATE),
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE
        ),
        "SVM": SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
    }


def train_and_evaluate_kfold(
    model: any, X: np.ndarray | pd.DataFrame, y: pd.Series, n_splits: int
) -> dict[str, float]:
    """
    Treina o modelo usando validação cruzada StratifiedKFold e calcula médias das métricas.

    Parâmetros:
        model (any): instância do classificador do scikit-learn
        X (np.ndarray | pd.DataFrame): features do dataset completo (normalizadas ou brutas)
        y (pd.Series): série contendo a variável alvo
        n_splits (int): número de dobras (k) para validação cruzada

    Retorna:
        dict[str, float]: dicionário com a média de acurácia, precisão, recall e f1-score
    """
    kfold = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE
    )
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision_weighted",
        "recall": "recall_weighted",
        "f1": "f1_weighted",
    }

    cv_results = cross_validate(
        model, X, y, cv=kfold, scoring=scoring, return_train_score=False
    )

    return {
        "acuracia": float(np.mean(cv_results["test_accuracy"])),
        "precisao": float(np.mean(cv_results["test_precision"])),
        "recall": float(np.mean(cv_results["test_recall"])),
        "f1": float(np.mean(cv_results["test_f1"])),
        "auc": np.nan,  # Fica vazio por convenção de K-fold no artigo original
    }


def train_and_evaluate_split(
    model: any,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict[str, float]:
    """
    Treina o modelo no conjunto de treino e calcula métricas sobre o conjunto de teste.

    Parâmetros:
        model (any): instância do classificador scikit-learn
        X_train (np.ndarray): features do conjunto de treino normalizadas
        X_test (np.ndarray): features do conjunto de teste normalizadas
        y_train (pd.Series): variável alvo de treino
        y_test (pd.Series): variável alvo de teste

    Retorna:
        dict[str, float]: dicionário com acurácia, precisão, recall, f1-score e auc-roc
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = (
        model.predict_proba(X_test)[:, 1]
        if hasattr(model, "predict_proba")
        else None
    )

    auc = (
        float(roc_auc_score(y_test, y_proba))
        if y_proba is not None
        else np.nan
    )

    return {
        "acuracia": float(accuracy_score(y_test, y_pred)),
        "precisao": float(precision_score(y_test, y_pred, average="weighted")),
        "recall": float(recall_score(y_test, y_pred, average="weighted")),
        "f1": float(f1_score(y_test, y_pred, average="weighted")),
        "auc": auc,
    }


def run_all_models(
    X: np.ndarray | pd.DataFrame,
    y: pd.Series,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: pd.Series,
    y_test: pd.Series,
) -> pd.DataFrame:
    """
    Treina e avalia todos os 7 modelos clássicos sob K-fold e sob Split.

    Parâmetros:
        X (np.ndarray | pd.DataFrame): features do dataset completo
        y (pd.Series): variável alvo do dataset completo
        X_train (np.ndarray): features de treino
        X_test (np.ndarray): features de teste
        y_train (pd.Series): alvo de treino
        y_test (pd.Series): alvo de teste

    Retorna:
        pd.DataFrame: DataFrame com os resultados agregados de todos os experimentos clássicos
    """
    models = get_classical_models()
    results = []

    for name, model in models.items():
        print(f"→ Avaliando modelo clássico: {name}")

        # 1. K-fold evaluation
        kfold_metrics = train_and_evaluate_kfold(model, X, y, n_splits=N_SPLITS)
        kfold_metrics["modelo"] = name
        kfold_metrics["metodo"] = "kfold"
        results.append(kfold_metrics)
        print(f"  - K-fold | Acurácia: {kfold_metrics['acuracia']:.4f}")

        # 2. Split evaluation
        split_metrics = train_and_evaluate_split(
            model, X_train, X_test, y_train, y_test
        )
        split_metrics["modelo"] = name
        split_metrics["metodo"] = "split"
        results.append(split_metrics)
        print(f"  - Split  | Acurácia: {split_metrics['acuracia']:.4f}")

    df_results = pd.DataFrame(results)
    # Reordenamento das colunas conforme especificações
    cols = ["modelo", "metodo", "acuracia", "precisao", "recall", "f1", "auc"]
    df_results = df_results[cols]

    return df_results
