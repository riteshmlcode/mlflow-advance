import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


# Load the iris dataset
iris = pd.read_csv('https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv')
X = iris.iloc[:,0:-1]
y = iris.iloc[:,-1]


# Split the data into a training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the parameters for the random forest model

max_depth = 10
n_estimators = 100

# Apply mlflow
mlflow.set_experiment('iris-rt')

with mlflow.start_run():
    # Create a random forest classifier
    rf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators)
    rf.fit(X_train, y_train)

    # Make predictions
    y_pred = rf.predict(X_test)

    # Log the model
    #mlflow.sklearn.log_model(rf, "random-forest-model")

    # Log the metrics
    accuracy = accuracy_score(y_test, y_pred)
    mlflow.log_metric('accuracy', accuracy)
    print(f"Accuracy: {accuracy}")  

    # Log the parameters
    mlflow.log_param('max_depth', max_depth)
    mlflow.log_param('n_estimators', n_estimators)
    # Create a confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", linewidths=.5, square = True, cmap = 'Blues');
    plt.ylabel('Actual label');
    plt.xlabel('Predicted label');
    plt.title('Confusion Matrix');

    # Save the confusion matrix
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")

    mlflow.log_artifact(__file__)
    # mlflow.sklearn.log_model(rf, "random-forest-model")
    mlflow.set_tag('model', 'RandomForest')
    mlflow.set_tag('author', 'xcode')

    # Logging Dataset
    train_df = pd.DataFrame(X_train)
    train_df['variety'] = y_train

    test_df = pd.DataFrame(X_test)
    test_df['variety'] = y_test

    train_df = mlflow.data.from_pandas(train_df)
    test_df = mlflow.data.from_pandas(test_df)

    mlflow.log_input(train_df, 'train')
    mlflow.log_input(test_df, 'validation')




