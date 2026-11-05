import logging

from src.domain.approval_rule import ApprovalRule
from src.repositories.csv_credit_repository import CsvCreditRepository
from src.services.credit_model import CreditModel
from src.use_cases.classify_credit_use_case import ClassifyCreditUseCase


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    use_case = ClassifyCreditUseCase(CsvCreditRepository(), CreditModel(), ApprovalRule())
    print(use_case.execute())


if __name__ == "__main__":
    main()
