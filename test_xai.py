import pandas as pd
import shap
from pycaret.classification import load_model, predict_model

print("1.Loading our model")

pipeline = load_model('ieee_fraud_model')

print("2.Trying some transactions for test")

trans = pd.read_csv('train_transaction.csv', nrows=50)
idents = pd.read_csv('train_identity.csv', nrows=50)
test_data = pd.merge(trans, idents, on='TransactionID', how='left')


test_data_input = test_data.drop(columns=['isFraud'])

print("3.Predicting...(classify)")

predictions = predict_model(pipeline, data=test_data_input)
print("\n--- Results ---")

print(predictions[['TransactionID', 'prediction_label', 'prediction_score']])

print("\n4.Reasons")


prep_data = pipeline[:-1].transform(test_data_input)
actual_model = pipeline.named_steps['trained_model']


explainer = shap.TreeExplainer(actual_model)
shap_values = explainer.shap_values(prep_data)

print("\n--- Finish Successfully ---")