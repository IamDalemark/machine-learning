import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from matplotlib import pyplot as plt
import seaborn as sns
import pickle
# data cleaning

data = pd.read_csv("diabetes_dataset.csv")

def preprocess_data(df):
    smoking_statuses = {
    'Never': 0,
    'Former': .5,
    'Current': 1,
}
    df = df.drop(columns=["age", "gender", "ethnicity", "education_level", "income_level", "employment_status",],axis=1)
    df["smoking_status"] = df["smoking_status"].map(smoking_statuses)
    return df

processed_data = preprocess_data(data)

# feature engineering
X = processed_data.drop(columns=["diabetes_risk_score", "diabetes_stage", "diagnosed_diabetes"],axis=1)
y = processed_data["diagnosed_diabetes"]

# ml prepocessing
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.25, random_state=16)
# model tune
print("starting scaling")
scalar = MinMaxScaler()
X_train = scalar.fit_transform(X_train)
X_test = scalar.transform(X_test)

def hyperTuneModel(X_train, Y_train):
    print("starting hypertuning")
    param_grid = {'penalty':['l2'],
    'C' : np.logspace(-4,4,20),
    'solver': ['lbfgs'],
    'max_iter'  : [100]
    }
    model = LogisticRegression()
    print("starting grid search")
    gridSearch = GridSearchCV(model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
    gridSearch.fit(X_train, Y_train)
    print("best param", gridSearch.best_params_)
    return gridSearch.best_estimator_
# evaluation

def evaluate_model(model, X_test, y_test):
    print("starting evaluation")
    prediction = model.predict(X_test)
    accuracy = accuracy_score(y_test, prediction)
    matrix = confusion_matrix(y_test, prediction)

    return accuracy, matrix


model = hyperTuneModel(X_train, Y_train)

accuracy, matrix = evaluate_model(model, X_test, Y_test)

print(f'Accuracy: {accuracy * 100:.2f} ')
print("Confusion Matrix")
print(matrix)


with open('diabetes_model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scalar, f)
print("--- nothing follows ---")