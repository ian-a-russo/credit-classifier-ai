import pandas as pd

class PandasDataProcessing:
    def __init__(self, data_frame: pd.DataFrame,):
        self._data_frame = data_frame.copy()

    def treat_rows(self: bool = False):
        self._data_frame = self._data_frame.dropna()

        return self._data_frame


        