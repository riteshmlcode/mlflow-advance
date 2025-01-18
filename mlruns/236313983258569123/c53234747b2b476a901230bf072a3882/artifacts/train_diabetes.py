from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import mlflow

df = pd.read_csv('https://raw.githubusercontent.com/npradaschnor/Pima-Indians-Diabetes-Dataset/master/diabetes.csv')
print(df.head())

# Splitting data into features and target
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Splitting data into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating the Random Forest Classifier
rf = RandomForestClassifier(random_state=42)

# Defining the grid search parameters for GridSearchCV
param_grid = {
    'n_estimators': [10, 50, 100],
    'max_depth': [None, 10, 20, 30]
}

# Applying GridSearchCV
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, n_jobs=-1, cv=5, verbose=2)

mlflow.set_experiment('diabetes')
with mlflow.start_run(run_name='Grid Search'):
    # Fitting the grid search   
    grid_search.fit(X_train, y_train)

    # Log all the Child Runs
    for i in range(len(grid_search.cv_results_['params'])):
        print(i)
        with mlflow.start_run(nested=True) as child:
        
            mlflow.log_params(grid_search.cv_results_['params'][i])
            mlflow.log_metric('accuracy', grid_search.cv_results_['mean_test_score'][i])

    # Displaying the best parameters
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    # Logging the best parameters and best score
    mlflow.log_params(best_params)
    mlflow.log_metric('best_score', best_score)

    # Logging the data
    train_df = X_train
    train_df['Outcome'] = y_train
    train_df = mlflow.data.from_pandas(train_df)
    mlflow.log_input(train_df, "Training Data")

    test_df = X_test
    test_df['Outcome'] = y_test
    test_df = mlflow.data.from_pandas(test_df)
    mlflow.log_input(test_df, "Training Data")

    # Logging the Source Code
    mlflow.log_artifact('train_diabetes.py')

    # Logging the model
    mlflow.sklearn.log_model(grid_search.best_estimator_, "model")

    print(f"Best parameters: {best_params}")
    print(f"Best score: {best_score}")



