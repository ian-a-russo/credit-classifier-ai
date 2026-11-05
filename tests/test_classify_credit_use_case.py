from pathlib import Path

import pandas as pd

from src.domain.approval_rule import ApprovalRule
from src.repositories.csv_credit_repository import CsvCreditRepository
from src.services.credit_model import CreditModel
from src.use_cases.classify_credit_use_case import ClassifyCreditUseCase

TRAINING_ROWS = [
    ("positive", 5000, "yes", "yes"),
    ("positive", 7000, "no", "yes"),
    ("negative", 6000, "yes", "no"),
    ("positive", 1500, "yes", "no"),
] * 5


def write_fixtures(directory: Path) -> CsvCreditRepository:
    columns = ["credit_history", "income", "employment", "property"]
    pd.DataFrame(TRAINING_ROWS, columns=columns).assign(age=30).to_csv(
        directory / "training.csv", index=False
    )
    pd.DataFrame(
        [("positive", 9000, "yes", "yes"), ("negative", 9000, "no", "no")], columns=columns
    ).assign(age=40).to_csv(directory / "testing.csv", index=False)
    return CsvCreditRepository(
        directory / "training.csv", directory / "testing.csv", directory / "out" / "result.csv"
    )


def test_execute_classifies_and_persists_results(tmp_path: Path) -> None:
    repository = write_fixtures(tmp_path)
    use_case = ClassifyCreditUseCase(repository, CreditModel(), ApprovalRule())

    results = use_case.execute()

    assert results["resultado"].tolist() == ["aceito", "negado"]
    assert (tmp_path / "out" / "result.csv").exists()
