import pandas as pd
from services.DecisionTreePipeline import DecisionTreePipeline

training_table = pd.read_csv('./data/training_data.csv')
training_table['approved'] = (
    (training_table['income'] > 3000) &
    (training_table['credit_history'] == 'positive')
).astype(int)

testing_table = pd.read_csv('./data/testing_data.csv')

categorical_columns = ['credit_history', 'employment', 'property']

pipeline = DecisionTreePipeline(training_table, testing_table, categorical_columns)
pipeline.encode_categoricals()

x = pipeline._training_table.drop(columns=["approved", "age"])
y = pipeline._training_table["approved"]

pipeline.train_model(x, y)
accuracy = pipeline.train_predict()
print("Acurácia:", accuracy)

decision_tree_test_forecast = pipeline.test()
formatted_result = pd.Series(decision_tree_test_forecast).map({1: "aceito", 0: "negado"})

result_table = pipeline._testing_table.copy()
result_table = result_table.reset_index(drop=True)
result_table["resultado"] = formatted_result

print(result_table)

result_table.to_csv("resultado_previsoes.csv", index=False)