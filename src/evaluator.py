"""
evaluator.py — Cálculo de métricas e persistência de resultados de modelos.

Este módulo implementa funções para calcular métricas de classificação binária,
persistir e atualizar os resultados dos experimentos no arquivo resultados.csv, e
comparar os desempenhos obtidos com os dados de referência do artigo original.

Referência: Khanam & Foo (2021), DOI: 10.1016/j.icte.2021.02.004
"""

# ── Stdlib ───────────────────────────────────────────────
from pathlib import Path

# ── Terceiros ────────────────────────────────────────────
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# ── Locais ───────────────────────────────────────────────
from src.config import RESULTS_DIR


def compute_metrics(
    y_true: any, y_pred: any, y_proba: any = None
) -> dict[str, float]:
    """
    Calcula acurácia, precisão ponderada, recall ponderado, F1-score ponderado e AUC-ROC.

    Parâmetros:
        y_true: rótulos reais das classes
        y_pred: predições de classes do modelo
        y_proba: probabilidade da classe positiva predita pelo modelo (opcional)

    Retorna:
        dict[str, float]: dicionário mapeando nome da métrica ao valor correspondente
    """
    auc = float(roc_auc_score(y_true, y_proba)) if y_proba is not None else np.nan
    return {
        "acuracia": float(accuracy_score(y_true, y_pred)),
        "precisao": float(precision_score(y_true, y_pred, average="weighted")),
        "recall": float(recall_score(y_true, y_pred, average="weighted")),
        "f1": float(f1_score(y_true, y_pred, average="weighted")),
        "auc": auc,
    }


def save_results(
    results: pd.DataFrame | list[dict[str, any]], filepath: Path = None
) -> None:
    """
    Salva ou atualiza os resultados no CSV resultados.csv sem apagar os dados antigos.

    Parâmetros:
        results (pd.DataFrame | list[dict]): resultados dos novos experimentos
        filepath (Path, opcional): caminho completo do arquivo csv de saída

    Retorna:
        None
    """
    if filepath is None:
        filepath = RESULTS_DIR / "resultados.csv"

    df_new = (
        pd.DataFrame(results) if isinstance(results, list) else results.copy()
    )

    if filepath.exists():
        df_existing = pd.read_csv(filepath)
        # Remove registros existentes correspondentes ao mesmo par (modelo, metodo)
        mask = ~(
            df_existing["modelo"].isin(df_new["modelo"])
            & df_existing["metodo"].isin(df_new["metodo"])
        )
        df_final = pd.concat([df_existing[mask], df_new], ignore_index=True)
    else:
        df_final = df_new

    # Cria diretório se não existir
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(filepath, index=False, float_format="%.4f")
    print(f"✓ Resultados salvos em: {filepath} ({len(df_final)} registros)")


def compare_with_reference(
    results_df: pd.DataFrame, reference: dict[str, any]
) -> pd.DataFrame:
    """
    Gera tabela comparativa entre os resultados de acurácia locais e os do artigo base.

    Parâmetros:
        results_df (pd.DataFrame): DataFrame de resultados locais
        reference (dict[str, any]): dicionário de referência contendo os dados do artigo

    Retorna:
        pd.DataFrame: DataFrame comparativo
    """
    ref_data = reference.get("resultados", {})
    comparison_rows = []

    # Dicionário mapeando os nomes locais aos nomes presentes no config do artigo
    mapping = {
        "Decision Tree": "Decision Tree",
        "KNN": "KNN",
        "Random Forest": "Random Forest",
        "Naive Bayes": "Naive Bayes",
        "AdaBoost": "AdaBoost",
        "Logistic Regression": "Logistic Regression",
        "SVM": "SVM",
        "NN-2 (400 épocas)": "Rede Neural (NN-2)",
    }

    for local_name, ref_name in mapping.items():
        # Busca acurácia local K-fold
        kfold_local = results_df[
            (results_df["modelo"] == local_name)
            & (results_df["metodo"] == "kfold")
        ]["acuracia"].values
        kfold_val = float(kfold_local[0]) if len(kfold_local) > 0 else np.nan

        # Busca acurácia local Split
        split_local = results_df[
            (results_df["modelo"] == local_name)
            & (results_df["metodo"] == "split")
        ]["acuracia"].values
        split_val = float(split_local[0]) if len(split_local) > 0 else np.nan

        # Coleta os valores correspondentes de referência
        ref_metrics = ref_data.get(ref_name, {})
        kfold_ref = ref_metrics.get("kfold", np.nan)
        split_ref = ref_metrics.get("split", np.nan)

        comparison_rows.append(
            {
                "Modelo": local_name,
                "Acurácia K-fold (Obtida)": kfold_val,
                "Acurácia K-fold (Artigo)": kfold_ref,
                "Diferença K-fold": (
                    kfold_val - kfold_ref
                    if not np.isnan(kfold_val) and not np.isnan(kfold_ref)
                    else np.nan
                ),
                "Acurácia Split (Obtida)": split_val,
                "Acurácia Split (Artigo)": split_ref,
                "Diferença Split": (
                    split_val - split_ref
                    if not np.isnan(split_val) and not np.isnan(split_ref)
                    else np.nan
                ),
            }
        )

    return pd.DataFrame(comparison_rows)
