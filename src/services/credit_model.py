import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src import config

FEATURES = config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES


class CreditModel:
    """Random Forest com pré-processamento embutido.

    Usar um único Pipeline garante que o encoder aprenda só com o treino e
    que o mesmo tratamento seja aplicado em validação e em dados novos.
    """

    def __init__(self, random_state: int = config.RANDOM_STATE) -> None:
        encoder = OneHotEncoder(handle_unknown="ignore")
        preprocessing = ColumnTransformer(
            [
                ("categorical", encoder, config.CATEGORICAL_FEATURES),
                ("numeric", "passthrough", config.NUMERIC_FEATURES),
            ]
        )
        self._pipeline = Pipeline(
            [
                ("preprocessing", preprocessing),
                ("classifier", RandomForestClassifier(random_state=random_state)),
            ]
        )

    def fit(self, applicants: pd.DataFrame, decisions: pd.Series) -> None:
        self._pipeline.fit(applicants[FEATURES], decisions)

    def evaluate(self, applicants: pd.DataFrame, decisions: pd.Series) -> float:
        return float(accuracy_score(decisions, self.predict(applicants)))

    def predict(self, applicants: pd.DataFrame) -> np.ndarray:
        return np.asarray(self._pipeline.predict(applicants[FEATURES]))
