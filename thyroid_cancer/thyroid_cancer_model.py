import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MaxAbsScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from matplotlib import pyplot as plt
import seaborn as sns
import pickle

# open

df = pd.read_csv("cancer_dataset.csv")

# preprocess data & vectorize document text
def preprocessCancer(df):
    df.drop("id", axis=1, inplace=True)
    df["cancer_type"] = np.where(df["cancer_type"] == "Thyroid_Cancer", 1, 0)
    return df["cancer_type"]

def preprocessResearch(df):
    vectorizer = TfidfVectorizer(max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(df["research_document"])
    return tfidf_matrix, vectorizer

# feature selection

def featureSelection(df):
    X, vectorizer = preprocessResearch(df)
    y = preprocessCancer(df)
    return X, y, vectorizer

X, y, vectorizer = featureSelection(df)

# scale & normalize
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.25, random_state=16)
scaler = MaxAbsScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# model & hypertuning

def hypertuneModel(X_train, y_train):
    model = LogisticRegression()
    param_grid = {'penalty':['l2'],
    'C' : np.logspace(-4,4,20),
    'solver': ['lbfgs'],
    'max_iter'  : [100]
    }
    gridsearch = GridSearchCV(model, param_grid=param_grid, n_jobs=-1)
    gridsearch.fit(X_train, y_train)
    print(gridsearch.best_params_)
    return gridsearch.best_estimator_

model = hypertuneModel(X_train, Y_train)
# evaluation

def evaluate(model, X_test, Y_test):
    prediction = model.predict(X_test)
    accuracy = accuracy_score(Y_test, prediction)
    matrix = confusion_matrix(Y_test, prediction)
    return accuracy, matrix

accuracy, matrix = evaluate(model, X_test, Y_test)

print("----Accuracy----")
print()
print(accuracy)
print("----matrix----")
print(matrix)

# save model ang dump

with open("thyroid_cancer_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("scaler.pkl", "wb") as f: 
    pickle.dump(scaler, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

