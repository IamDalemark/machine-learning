import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from matplotlib import pyplot as plt
import seaborn as sns
# Data Cleaning
# def checkNan(df, name):
#     print(df.isna().sum())
#     print("test " + name)

df = pd.read_csv("student_exam_scores.csv")
def preprocess_data(df):
    df.drop("student_id", axis=1, inplace=True)
    df["exam_score"] = np.where(df["exam_score"] >= 35, 1, 0)
    return df

data = preprocess_data(df)
# Feature Engineering
# y is the answer while x is the question
X = data.drop(columns="exam_score" )
y = data["exam_score"]


X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.25, random_state=16)

#Ml preprocessing
scalar = MinMaxScaler()
X_train = scalar.fit_transform(X_train)
X_test = scalar.transform(X_test)

# tune
def hypertunemodel(X_train, Y_train):
    params = {
        "n_neighbors": range(1,21),
        "metric": ["euclidean", "manhattan", "minkowski"],
        "weights": ["uniform", "distance"]
    }
    model = KNeighborsClassifier()
    gridSearch = GridSearchCV(model, param_grid=params, cv=5, n_jobs=1)
    gridSearch.fit(X_train, Y_train)
    return gridSearch.best_estimator_
# checkNan(data, "test")
#Predictions and evalutaion
def evaluate_model(model, X_test, y_test):
    prediction = model.predict(X_test)
    accuracy = accuracy_score(y_test, prediction)
    matrix = confusion_matrix(y_test, prediction)

    return accuracy, matrix

model = hypertunemodel(X_train, Y_train)

accuracy, matrix = evaluate_model(model, X_test, Y_test)

print(f'Accuracy: {accuracy * 100:.2f} ')
print("Confusion Matrix")
print(matrix)
# evaluation

def plot_confusion_matrix(matrix):
    plt.figure(figsize=(7,6))
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted Negative', 'Predicted Positive'], yticklabels=['Actual Negative', 'Actual Positive'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.show()

plot_confusion_matrix(matrix)