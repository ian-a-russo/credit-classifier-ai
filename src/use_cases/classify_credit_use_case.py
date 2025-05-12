import pandas as pd
from src.services.decision_tree_pipeline import DecisionTreePipeline

class ClassifyCreditUseCase:
    def execute(self):
        training_table = pd.read_csv('src/data/training_data.csv')
        training_table['approved'] = (
            (training_table['income'] > 3000) &
            (training_table['credit_history'] == 'positive')
        ).astype(int)

        testing_table = pd.read_csv('src/data/testing_data.csv')

        categorical_columns = ['credit_history', 'employment', 'property']

        self._pipeline = DecisionTreePipeline(training_table, testing_table, categorical_columns)
        self._pipeline.encode_categoricals()

        x = self._pipeline._training_table.drop(columns=["approved", "age"])
        y = self._pipeline._training_table["approved"]

        self._pipeline.train_model(x, y)
        self._pipeline.train_predict()

        decision_tree_test_forecast = self._pipeline.test()

        self.format_and_save_result(decision_tree_test_forecast)

        
    def format_and_save_result(self, dataframe: pd.DataFrame):
        formatted_result = pd.Series(dataframe).map({1: "aceito", 0: "negado"})

        result_table = self._pipeline._testing_table.copy()
        result_table = result_table.reset_index(drop=True)
        result_table["resultado"] = formatted_result

        print(result_table)

        result_table.to_csv("src/data/results/predictions_credit_result.csv", index=False)