import logging

import pandas as pd
from sklearn.model_selection import train_test_split

from src import config
from src.domain.approval_rule import ApprovalRule
from src.repositories.csv_credit_repository import CsvCreditRepository
from src.services.credit_model import CreditModel

logger = logging.getLogger(__name__)


class ClassifyCreditUseCase:
    def __init__(
        self,
        repository: CsvCreditRepository,
        model: CreditModel,
        rule: ApprovalRule,
    ) -> None:
        self._repository = repository
        self._model = model
        self._rule = rule

    def execute(self) -> pd.DataFrame:
        self._train()
        results = self._classify(self._repository.load_testing())
        self._repository.save_results(results)
        return results

    def _train(self) -> None:
        history = self._repository.load_training()
        decisions = self._rule.label(history)
        x_train, x_valid, y_train, y_valid = train_test_split(
            history,
            decisions,
            test_size=config.VALIDATION_SIZE,
            random_state=config.RANDOM_STATE,
            stratify=decisions,
        )
        self._model.fit(x_train, y_train)
        logger.info("Acurácia na validação: %.4f", self._model.evaluate(x_valid, y_valid))

    def _classify(self, applicants: pd.DataFrame) -> pd.DataFrame:
        decisions = pd.Series(self._model.predict(applicants), index=applicants.index)
        return applicants.assign(resultado=self._rule.describe(decisions))
