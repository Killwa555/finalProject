import pandas as pd
from pycaret.classification import setup, compare_models, save_model

print("Reading data files...")

transactions = pd.read_csv('train_transaction.csv')
identities = pd.read_csv('train_identity.csv')

print("1.Merging tables of data")

dataset = pd.merge(transactions, identities, on='TransactionID', how='left')


data_sample = dataset.sample(frac=0.05, random_state=42)

print("3.PyCaret Setup")

ai_setup = setup(data=data_sample, target='isFraud', fix_imbalance=True, session_id=123)

print("4.Comparing wich algorithm is the best")

best_ai_model = compare_models(n_select=1)

print("5.Model Saving...")

save_model(best_ai_model, 'ieee_fraud_model')

print("congratulation your ai is traind and saved successfully")