"""
visualizer.py — Geração de visualizações e gráficos do projeto.

Este módulo implementa a geração de gráficos de barras comparativos para acurácia
e F1-score, plotagem de matriz de confusão e curva ROC para a rede neural, curvas
de aprendizado (loss/accuracy) e importância de features com Random Forest.

Referência: Khanam & Foo (2021), DOI: 10.1016/j.icte.2021.02.004
"""

# ── Stdlib ───────────────────────────────────────────────
from pathlib import Path

# ── Terceiros ────────────────────────────────────────────
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve

# ── Locais ───────────────────────────────────────────────
from src.evaluator import compare_with_reference


def plot_accuracy_comparison(results_df: pd.DataFrame, save_path: Path) -> None:
    """
    Gráfico de barras comparativo da acurácia K-fold vs Split por modelo.

    Parâmetros:
        results_df (pd.DataFrame): DataFrame com os resultados salvos
        save_path (Path): caminho onde o gráfico PNG será salvo
    """
    plt.style.use("seaborn-v0_8-whitegrid")
    df_pivot = results_df.pivot(
        index="modelo", columns="metodo", values="acuracia"
    )

    # Reordena para destacar modelos clássicos e redes neurais de forma legível
    fig, ax = plt.subplots(figsize=(12, 6))
    df_pivot.plot(kind="bar", ax=ax, color=["#1A56A0", "#E65100"])

    ax.set_title(
        "Comparação de Acurácia: K-fold vs Split",
        fontsize=14,
        fontweight="bold",
        pad=16,
    )
    ax.set_xlabel("Modelos", fontsize=11)
    ax.set_ylabel("Acurácia", fontsize=11)
    ax.set_ylim(0.5, 1.0)
    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"✓ Gráfico salvo em: {save_path}")


def plot_f1_comparison(results_df: pd.DataFrame, save_path: Path) -> None:
    """
    Gráfico de barras comparativo do F1-score K-fold vs Split por modelo.

    Parâmetros:
        results_df (pd.DataFrame): DataFrame com os resultados salvos
        save_path (Path): caminho onde o gráfico PNG será salvo
    """
    plt.style.use("seaborn-v0_8-whitegrid")
    df_pivot = results_df.pivot(index="modelo", columns="metodo", values="f1")

    fig, ax = plt.subplots(figsize=(12, 6))
    df_pivot.plot(kind="bar", ax=ax, color=["#2E7D32", "#6A1B9A"])

    ax.set_title(
        "Comparação de F1-Score: K-fold vs Split",
        fontsize=14,
        fontweight="bold",
        pad=16,
    )
    ax.set_xlabel("Modelos", fontsize=11)
    ax.set_ylabel("F1-Score", fontsize=11)
    ax.set_ylim(0.5, 1.0)
    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"✓ Gráfico salvo em: {save_path}")


def plot_confusion_matrix(
    y_true: any, y_pred: any, model_name: str, save_path: Path
) -> None:
    """
    Matriz de confusão para o modelo de melhor acurácia obtida.

    Parâmetros:
        y_true: rótulos reais das classes
        y_pred: predições do modelo
        model_name (str): nome do modelo a ser exibido
        save_path (Path): caminho onde o gráfico PNG será salvo
    """
    plt.style.use("seaborn-v0_8-whitegrid")
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax,
        cbar=False,
        annot_kws={"fontsize": 14, "fontweight": "bold"},
    )

    ax.set_title(
        f"Matriz de Confusão — {model_name}",
        fontsize=14,
        fontweight="bold",
        pad=16,
    )
    ax.set_xlabel("Predito", fontsize=11)
    ax.set_ylabel("Real", fontsize=11)
    ax.set_xticklabels(["Sem Diabetes", "Com Diabetes"])
    ax.set_yticklabels(["Sem Diabetes", "Com Diabetes"])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"✓ Gráfico salvo em: {save_path}")


def plot_roc_curve(
    y_true: any, y_proba: any, model_name: str, save_path: Path
) -> None:
    """
    Plota a curva ROC para a probabilidade predita e calcula sua AUC.

    Parâmetros:
        y_true: rótulos reais das classes
        y_proba: probabilidade da classe positiva
        model_name (str): nome do modelo
        save_path (Path): caminho onde o gráfico PNG será salvo
    """
    plt.style.use("seaborn-v0_8-whitegrid")
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = float(np.mean(y_proba))  # apenas placeholder se não houver ROC real, mas temos
    from sklearn.metrics import auc

    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        fpr,
        tpr,
        color="#1A56A0",
        lw=2,
        label=f"Curva ROC (AUC = {roc_auc:.4f})",
    )
    ax.plot(
        [0, 1],
        [0, 1],
        color="#E65100",
        lw=2,
        linestyle="--",
        label="Classificador Aleatório",
    )

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_title(
        f"Curva ROC — {model_name}", fontsize=14, fontweight="bold", pad=16
    )
    ax.set_xlabel("Taxa de Falso Positivo (FPR)", fontsize=11)
    ax.set_ylabel("Taxa de Verdadeiro Positivo (TPR)", fontsize=11)
    ax.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"✓ Gráfico salvo em: {save_path}")


def plot_learning_curves(history_dict: dict[str, dict], save_path: Path) -> None:
    """
    Curvas de loss e accuracy por época para as 3 arquiteturas de NN (200/400/800 épocas).
    Foca nas configurações de 400 épocas para comparabilidade das arquiteturas.

    Parâmetros:
        history_dict (dict): histórico de todos os experimentos das redes neurais
        save_path (Path): caminho onde o gráfico PNG será salvo
    """
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    colors = {"NN-1": "#1A56A0", "NN-2": "#2E7D32", "NN-3": "#E65100"}

    for name in ["NN-1 (400 épocas)", "NN-2 (400 épocas)", "NN-3 (400 épocas)"]:
        short_name = name.split(" ")[0]
        if name in history_dict:
            h = history_dict[name]
            epochs_range = range(1, len(h["loss"]) + 1)

            # Subplot 1: Loss
            ax1.plot(
                epochs_range,
                h["loss"],
                label=f"{short_name} Treino",
                color=colors[short_name],
                linestyle="--",
            )
            if "val_loss" in h:
                ax1.plot(
                    epochs_range,
                    h["val_loss"],
                    label=f"{short_name} Validação",
                    color=colors[short_name],
                )

            # Subplot 2: Accuracy
            ax2.plot(
                epochs_range,
                h["accuracy"],
                label=f"{short_name} Treino",
                color=colors[short_name],
                linestyle="--",
            )
            if "val_accuracy" in h:
                ax2.plot(
                    epochs_range,
                    h["val_accuracy"],
                    label=f"{short_name} Validação",
                    color=colors[short_name],
                )

    ax1.set_title(
        "Curvas de Perda (Loss)", fontsize=14, fontweight="bold", pad=16
    )
    ax1.set_xlabel("Épocas", fontsize=11)
    ax1.set_ylabel("Loss", fontsize=11)
    ax1.legend()

    ax2.set_title(
        "Curvas de Acurácia (Accuracy)",
        fontsize=14,
        fontweight="bold",
        pad=16,
    )
    ax2.set_xlabel("Épocas", fontsize=11)
    ax2.set_ylabel("Acurácia", fontsize=11)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"✓ Gráfico salvo em: {save_path}")


def plot_feature_importance(
    model: any, feature_names: list[str], save_path: Path
) -> None:
    """
    Plota o gráfico de importância de features para o classificador Random Forest.

    Parâmetros:
        model: classificador Random Forest treinado
        feature_names (list): lista com os nomes das features correspondentes
        save_path (Path): caminho onde o gráfico PNG será salvo
    """
    plt.style.use("seaborn-v0_8-whitegrid")
    importances = model.feature_importances_
    indices = np.argsort(importances)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(
        range(len(indices)), importances[indices], color="#2E7D32", align="center"
    )
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices], fontsize=11)

    ax.set_title(
        "Importância das Features (Random Forest)",
        fontsize=14,
        fontweight="bold",
        pad=16,
    )
    ax.set_xlabel("Importância Relativa", fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"✓ Gráfico salvo em: {save_path}")


def plot_article_comparison(
    results_df: pd.DataFrame, reference: dict[str, any], save_path: Path
) -> None:
    """
    Plota uma tabela comparativa com os resultados de acurácia locais vs originais do artigo.

    Parâmetros:
        results_df (pd.DataFrame): DataFrame de resultados locais
        reference (dict): dicionário de dados de referência do artigo
        save_path (Path): caminho onde a imagem PNG da tabela será salva
    """
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis("off")
    ax.axis("tight")

    df_comp = compare_with_reference(results_df, reference)

    # Formatação dos dados para exibição textual na tabela
    df_disp = df_comp.copy()
    for col in df_disp.columns:
        if col != "Modelo":
            df_disp[col] = df_disp[col].apply(
                lambda x: f"{x * 100:.2f}%" if not pd.isna(x) else "-"
            )

    table = ax.table(
        cellText=df_disp.values,
        colLabels=df_disp.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.1, 1.8)

    # Estilização da tabela matplotlib
    for key, cell in table.get_celld().items():
        cell.set_edgecolor("#D3D3D3")
        if key[0] == 0:  # Cabeçalho
            cell.set_text_props(weight="bold", color="white")
            cell.set_facecolor("#1A56A0")
        else:  # Linhas de dados
            if key[0] % 2 == 0:
                cell.set_facecolor("#F0F8FF")

    ax.set_title(
        "Comparação Lado a Lado: Nossos Resultados vs Artigo",
        fontsize=14,
        fontweight="bold",
        pad=16,
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"✓ Gráfico salvo em: {save_path}")
