import pandas as pd

DECISION_LABELS = {1: "aceito", 0: "negado"}


class ApprovalRule:
    """Regra de negócio usada para rotular o histórico de treino.

    O dataset de treino não traz o desfecho real, então ele é derivado desta regra.
    """

    def __init__(self, min_income: float = 3000, required_history: str = "positive") -> None:
        self._min_income = min_income
        self._required_history = required_history

    def label(self, applicants: pd.DataFrame) -> pd.Series:
        has_income = applicants["income"] > self._min_income
        has_history = applicants["credit_history"] == self._required_history
        return (has_income & has_history).astype(int)

    @staticmethod
    def describe(decisions: pd.Series) -> pd.Series:
        return decisions.map(DECISION_LABELS)
