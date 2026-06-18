"""
data_loader.py — Carregamento e validação do dataset Pima Indians Diabetes.

Este módulo contém funções para carregar o dataset de diabetes a partir de um
caminho de arquivo específico, validar seu formato e dimensões, e gerar um
resumo estatístico básico incluindo a contagem de zeros.

Referência: Khanam & Foo (2021), DOI: 10.1016/j.icte.2021.02.004
"""

# ── Stdlib ───────────────────────────────────────────────
from pathlib import Path

# ── Terceiros ────────────────────────────────────────────
import pandas as pd

# ── Locais ───────────────────────────────────────────────
from src.config import ZERO_COLS


def load_dataset(filepath: Path) -> pd.DataFrame:
    """
    Carrega o dataset CSV e valida sua integridade de formato e existência.

    Parâmetros:
        filepath (Path): caminho para o arquivo diabetes.csv

    Retorna:
        pd.DataFrame: dataset carregado com 768 linhas × 9 colunas

    Lança:
        FileNotFoundError: se o arquivo não existir ou não for um arquivo válido.
        ValueError: se o número de colunas for diferente de 9.

    Exemplo:
        >>> from pathlib import Path
        >>> df = load_dataset(Path("data/diabetes.csv"))
        >>> print(df.shape)
        (768, 9)
    """
    if not filepath.is_file():
        raise FileNotFoundError(f"✗ Arquivo não encontrado em {filepath}")

    df = pd.read_csv(filepath)

    if df.shape[1] != 9:
        raise ValueError(
            f"✗ O número de colunas é diferente de 9 (encontrado: {df.shape[1]})"
        )

    print(f"✓ Dataset carregado: {df.shape[0]} registros")
    return df


def summarize_dataset(df: pd.DataFrame) -> None:
    """
    Exibe resumo estatístico, tipos de dados e contagem de zeros por coluna.

    Parâmetros:
        df (pd.DataFrame): DataFrame do pandas com os dados a serem resumidos

    Retorna:
        None
    """
    print("→ Resumo do Dataset:")
    print(f"  - Dimensões: {df.shape[0]} linhas × {df.shape[1]} colunas")

    print("\n→ Tipos de dados por coluna:")
    for col, dtype in df.dtypes.items():
        print(f"  - {col}: {dtype}")

    print("\n→ Estatísticas descritivas:")
    print(df.describe().to_string())

    print("\n→ Contagem de valores zero por coluna:")
    for col in df.columns:
        num_zeros = (df[col] == 0).sum()
        if col in ZERO_COLS:
            status = "⚠ (Suspeito)" if num_zeros > 0 else "✓ (Válido)"
            print(f"  - {col}: {num_zeros} zeros {status}")
        else:
            print(f"  - {col}: {num_zeros} zeros (Válido)")
