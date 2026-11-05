from pathlib import Path

DATA_DIR = Path("data")
TRAINING_FILE = DATA_DIR / "training_data.csv"
TESTING_FILE = DATA_DIR / "testing_data.csv"
RESULTS_FILE = DATA_DIR / "results" / "predictions_credit_result.csv"

NUMERIC_FEATURES = ["income"]
CATEGORICAL_FEATURES = ["credit_history", "employment", "property"]
TARGET = "approved"

# Fixo para que treino/validação e resultados sejam reproduzíveis
RANDOM_STATE = 42
VALIDATION_SIZE = 0.2
