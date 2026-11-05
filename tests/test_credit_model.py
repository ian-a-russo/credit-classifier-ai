import pandas as pd

from src.services.credit_model import CreditModel


def build_applicants() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "income": [5000.0, 6000.0, 1000.0, 1500.0] * 5,
            "credit_history": ["positive", "positive", "negative", "negative"] * 5,
            "employment": ["yes", "no", "yes", "no"] * 5,
            "property": ["yes", "no", "no", "yes"] * 5,
        }
    )


def build_decisions() -> pd.Series:
    return pd.Series([1, 1, 0, 0] * 5)


def test_model_learns_separable_pattern() -> None:
    model = CreditModel()
    model.fit(build_applicants(), build_decisions())

    assert model.evaluate(build_applicants(), build_decisions()) == 1.0


def test_predict_ignores_unseen_categories() -> None:
    model = CreditModel()
    model.fit(build_applicants(), build_decisions())
    unseen = build_applicants().head(1).assign(employment="retired")

    assert len(model.predict(unseen)) == 1


def test_predict_ignores_extra_columns() -> None:
    model = CreditModel()
    model.fit(build_applicants(), build_decisions())
    with_age = build_applicants().head(2).assign(age=[30, 40])

    assert len(model.predict(with_age)) == 2
