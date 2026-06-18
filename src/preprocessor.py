"""
preprocessor.py — Pipeline de pré-processamento do dataset Pima Indians.

Este módulo implementa o pipeline de pré-processamento de dados para o modelo de
predição de diabetes, incluindo substituição de valores impossíveis (zeros) por
NaN, imputação pela mediana, remoção de outliers por IQR, seleção de features,
divisão treino/teste estratificada e normalização MinMax.

Referência: Khanam & Foo (2021), DOI: 10.1016/j.icte.2021.02.004
"""

# ── Terceiros ────────────────────────────────────────────
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# ── Locais ───────────────────────────────────────────────
from src.config import (
    FEATURE_COLS,
    IQR_FACTOR,
    RANDOM_STATE,
    TARGET_COL,
    TEST_SIZE,
    ZERO_COLS,
)


def replace_zeros_with_nan(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    Substitui valores zero por NaN nas colunas especificadas.

    Parâmetros:
        df (pd.DataFrame): DataFrame original do dataset
        cols (list[str]): lista com nomes das colunas de interesse

    Retorna:
        pd.DataFrame: cópia do DataFrame com os zeros substituídos por NaN
    """
    df_copy = df.copy()
    df_copy[cols] = df_copy[cols].replace(0, np.nan)
    return df_copy


def impute_with_median(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preenche valores ausentes (NaN) com a mediana de cada coluna.

    Parâmetros:
        df (pd.DataFrame): DataFrame contendo valores NaN

    Retorna:
        pd.DataFrame: DataFrame com valores nulos preenchidos pela mediana
    """
    df_copy = df.copy()
    # fillna com a mediana calculada por coluna
    df_copy = df_copy.fillna(df_copy.median())
    return df_copy


def remove_outliers_iqr(df: pd.DataFrame, factor: float = 1.5) -> pd.DataFrame:
    """
    Remove outliers de todas as colunas usando o método Interquartile Range (IQR).

    Parâmetros:
        df (pd.DataFrame): DataFrame após imputação de nulos
        factor (float): multiplicador do IQR para limites superior/inferior (padrão: 1.5)

    Retorna:
        pd.DataFrame: DataFrame com outliers removidos
    """
    q1 = df.quantile(0.25)
    q3 = df.quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr

    # Filtra linhas onde nenhuma coluna é outlier
    mask = ~((df < lower_bound) | (df > upper_bound)).any(axis=1)
    df_clean = df[mask].copy()

    print(f"⚠ Aviso: {df.shape[0] - df_clean.shape[0]} outliers removidos")
    return df_clean


def select_features(
    df: pd.DataFrame, feature_cols: list[str], target_col: str
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Separa o DataFrame em conjunto de features (X) e rótulo alvo (y).

    Parâmetros:
        df (pd.DataFrame): DataFrame limpo
        feature_cols (list[str]): lista com colunas de features a serem mantidas
        target_col (str): nome da coluna alvo

    Retorna:
        tuple[pd.DataFrame, pd.Series]: (X, y)
    """
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    return X, y


def split_dataset(
    X: pd.DataFrame, y: pd.Series, test_size: float, random_state: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Divide o dataset em conjuntos de treino e teste de forma estratificada.

    Parâmetros:
        X (pd.DataFrame): DataFrame de features
        y (pd.Series): Série de alvo
        test_size (float): tamanho relativo do conjunto de teste (ex: 0.15)
        random_state (int): semente para reprodutibilidade (ex: 42)

    Retorna:
        tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
            (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    print(
        f"✓ Treino: {X_train.shape[0]} registros "
        f"({X_train.shape[0] / X.shape[0] * 100:.1f}%)"
    )
    print(
        f"✓ Teste:  {X_test.shape[0]} registros "
        f"({X_test.shape[0] / X.shape[0] * 100:.1f}%)"
    )

    return X_train, X_test, y_train, y_test


def normalize_features(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, MinMaxScaler]:
    """
    Aplica MinMaxScaler no intervalo [0, 1] ajustando o scaler apenas no treino.

    Parâmetros:
        X_train (pd.DataFrame): conjunto de treino
        X_test (pd.DataFrame): conjunto de teste

    Retorna:
        tuple[np.ndarray, np.ndarray, MinMaxScaler]:
            (X_train_scaled, X_test_scaled, scaler)
    """
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def run_full_pipeline(df: pd.DataFrame) -> dict:
    """
    Executa o pipeline completo de pré-processamento de dados.

    Parâmetros:
        df (pd.DataFrame): DataFrame original

    Retorna:
        dict: Dicionário contendo:
            - 'X_train': array numpy das features de treino normalizadas
            - 'X_test': array numpy das features de teste normalizadas
            - 'y_train': série do pandas das classes de treino
            - 'y_test': série do pandas das classes de teste
            - 'scaler': MinMaxScaler ajustado no conjunto de treino
    """
    # 1. Substitui zeros por NaN
    df_nan = replace_zeros_with_nan(df, ZERO_COLS)

    # 2. Imputação de nulos pela mediana
    df_imputed = impute_with_median(df_nan)

    # 3. Remoção de outliers
    df_clean = remove_outliers_iqr(df_imputed, IQR_FACTOR)

    # 4. Seleção de features
    X, y = select_features(df_clean, FEATURE_COLS, TARGET_COL)

    # 5. Split de treino e teste
    X_train, X_test, y_train, y_test = split_dataset(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # 6. Normalização MinMaxScaler
    X_train_scaled, X_test_scaled, scaler = normalize_features(X_train, X_test)

    print(
        f"✓ Pré-processamento concluído: "
        f"{X_train_scaled.shape[0]} treino | {X_test_scaled.shape[0]} teste"
    )

    return {
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
    }
