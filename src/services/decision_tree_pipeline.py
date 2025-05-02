import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class DecisionTreePipeline:
    def __init__(self, training_table: pd.DataFrame, testing_table: pd.DataFrame, categorical_columns: list[str]):
        self._model = RandomForestClassifier()
        self._training_table = training_table.copy()
        self._testing_table = testing_table.copy()
        self._categorical_columns = categorical_columns
        self._encoders = {}
        self._target_encoder = LabelEncoder()  # novo

    def encode_categoricals(self):
        for column in self._categorical_columns:
            encoder = LabelEncoder()
            self._training_table[column] = encoder.fit_transform(self._training_table[column])
            self._testing_table[column] = encoder.transform(self._testing_table[column])
            self._encoders[column] = encoder

    def train_model(self, x: pd.DataFrame, y: pd.Series):
        self._x_train, self._x_test, self._y_train, self._y_test = train_test_split(
            x, y, test_size=0.2, random_state=42
        )
        self._model.fit(self._x_train, self._y_train)


    def train_predict(self):
        prediction = self._model.predict(self._x_test)
        accuracy = accuracy_score(self._y_test, prediction)

        print("Acurácia:", accuracy)

    def test(self):
        x_real_test = self._testing_table[self._x_train.columns]
        return self._model.predict(x_real_test)
