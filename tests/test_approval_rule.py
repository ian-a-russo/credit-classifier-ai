import pandas as pd

from src.domain.approval_rule import ApprovalRule


def build_applicants() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "income": [5000.0, 5000.0, 2000.0, 3000.0],
            "credit_history": ["positive", "negative", "positive", "positive"],
        }
    )


def test_label_approves_only_high_income_with_positive_history() -> None:
    labels = ApprovalRule().label(build_applicants())

    assert labels.tolist() == [1, 0, 0, 0]


def test_label_requires_income_strictly_above_minimum() -> None:
    labels = ApprovalRule(min_income=3000).label(build_applicants())

    assert labels.iloc[3] == 0


def test_describe_maps_decisions_to_portuguese_labels() -> None:
    described = ApprovalRule.describe(pd.Series([1, 0]))

    assert described.tolist() == ["aceito", "negado"]
