from pathlib import Path

import pandas as pd

from src import config


class CsvCreditRepository:
    """Isola a leitura e a escrita dos arquivos CSV do restante da aplicação."""

    def __init__(
        self,
        training_file: Path = config.TRAINING_FILE,
        testing_file: Path = config.TESTING_FILE,
        results_file: Path = config.RESULTS_FILE,
    ) -> None:
        self._training_file = training_file
        self._testing_file = testing_file
        self._results_file = results_file

    def load_training(self) -> pd.DataFrame:
        return pd.read_csv(self._training_file)

    def load_testing(self) -> pd.DataFrame:
        return pd.read_csv(self._testing_file)

    def save_results(self, results: pd.DataFrame) -> None:
        self._results_file.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(self._results_file, index=False)
