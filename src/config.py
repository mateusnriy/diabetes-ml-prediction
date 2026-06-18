"""
config.py — Constantes e configurações globais do projeto.

Este módulo centraliza todas as configurações de caminhos, parâmetros
do dataset, hiperparâmetros de pré-processamento e treinamento, bem
como os resultados de referência do artigo base.

Referência: Khanam & Foo (2021), DOI: 10.1016/j.icte.2021.02.004
"""

# ── Stdlib ───────────────────────────────────────────────
from pathlib import Path

# ── Caminhos ────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"
GRAFICOS_DIR = RESULTS_DIR / "graficos"

# ── Dataset ─────────────────────────────────────────────
DATA_FILE = DATA_DIR / "diabetes.csv"
TARGET_COL = "Outcome"
FEATURE_COLS = ["Glucose", "BMI", "Insulin", "Pregnancies", "Age"]
ZERO_COLS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
CONTINUOUS_COLS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI",
                   "DiabetesPedigreeFunction", "Age"]

# ── Pré-processamento ────────────────────────────────────
IQR_FACTOR = 1.5
TEST_SIZE = 0.15
N_SPLITS = 7

# ── Treinamento ──────────────────────────────────────────
RANDOM_STATE = 42
EPOCHS_LIST = [200, 400, 800]
LEARNING_RATE = 0.01
BATCH_SIZE = 32

# ── Artigo de referência ─────────────────────────────────
REFERENCE = {
    "titulo": "A comparison of machine learning algorithms for diabetes prediction",
    "autores": "Khanam, J.J. & Foo, S.Y.",
    "ano": 2021,
    "doi": "10.1016/j.icte.2021.02.004",
    "resultados": {
        "Decision Tree": {"kfold": 0.7424, "split": 0.7314},
        "Random Forest": {"kfold": 0.7496, "split": 0.7714},
        "Naive Bayes": {"kfold": 0.7553, "split": 0.7828},
        "Logistic Regression": {"kfold": 0.7682, "split": 0.7885},
        "KNN": {"kfold": 0.7510, "split": 0.7942},
        "AdaBoost": {"kfold": 0.7396, "split": 0.7942},
        "SVM": {"kfold": 0.7682, "split": 0.7771},
        "Rede Neural (NN-2)": {"kfold": 0.7600, "split": 0.8857},
    },
}
