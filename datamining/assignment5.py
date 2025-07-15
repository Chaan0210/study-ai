import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.inspection import permutation_importance
import time

def load_preprocess_data(path, test_size=0.2, random_state=1, encoding='label'):
    df = pd.read_csv(path)
    if encoding == 'label':
        categorical_columns = ['sex', 'class', 'deck', 'embark_town', 'alone']
        for col in categorical_columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
    elif encoding == 'onehot':
        categorical_columns = ['sex', 'class', 'deck', 'embark_town', 'alone']
        df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
    else:
        raise ValueError("Unknown encoding type.")
    
    X = df.drop('survived', axis=1)
    y = df['survived']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    return X_train, y_train, X_test, y_test

def fit_model(X_train, y_train, random_state=1):
    param_grid = {
        'n_estimators': [25, 50, 100, 125, 150, 200],
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 2, 4, 6, 8, 10, 15, 20],
        'min_samples_split': [2, 3, 4, 5, 6, 8, 10, 15, 20],
        'min_samples_leaf': [2, 4, 5, 6, 8, 10],
        'max_features': ['sqrt', 'log2', None]
    }

    rf = RandomForestClassifier(random_state=random_state)
    grid_search = GridSearchCV(
        rf, 
        param_grid, 
        cv=5, 
        scoring='accuracy', 
        verbose=1, 
        n_jobs=-1, 
        error_score='raise'
    )
    grid_search.fit(X_train, y_train)

    print(f"Best params: {grid_search.best_params_}")
    print(f"CV Best Accuracy: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    return test_accuracy

def compare_feature_importances(model, X_test, y_test, feature_names):
    mdi_importances = model.feature_importances_
    
    perm_result = permutation_importance(
        model, X_test, y_test, n_repeats=10, random_state=1, n_jobs=-1
    )
    perm_importances = perm_result.importances_mean
    
    indices = np.argsort(mdi_importances)[::-1]
    sorted_features = np.array(feature_names)[indices]
    sorted_mdi = mdi_importances[indices]
    sorted_perm = perm_importances[indices]
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    
    # MDI plot
    ax[0].bar(range(len(sorted_features)), sorted_mdi, align='center')
    ax[0].set_xticks(range(len(sorted_features)))
    ax[0].set_xticklabels(sorted_features, rotation=45, ha='right')
    ax[0].set_title("MDI Feature Importances")
    ax[0].set_xlabel("Features")
    ax[0].set_ylabel("Importance")
    
    # Permutation plot
    ax[1].bar(range(len(sorted_features)), sorted_perm, align='center')
    ax[1].set_xticks(range(len(sorted_features)))
    ax[1].set_xticklabels(sorted_features, rotation=45, ha='right')
    ax[1].set_title("Permutation Feature Importances")
    ax[1].set_xlabel("Features")
    ax[1].set_ylabel("Importance")
    
    plt.tight_layout()
    plt.show()


def main():
    start_time = time.perf_counter()
    file_path = './titanic.csv'
    encoding_methods = ['label', 'onehot']
    results = {}

    for encoding in encoding_methods:
        print(f"----- Encoding Method: {encoding} -----")
        X_train, y_train, X_test, y_test = load_preprocess_data(file_path, test_size=0.2, random_state=1, encoding=encoding)
        best_rf = fit_model(X_train, y_train, random_state=1)
        test_acc = evaluate_model(best_rf, X_test, y_test)
        results[encoding] = {
            'model': best_rf,
            'accuracy': test_acc,
            'features': X_train.columns,
            'X_test': X_test,
            'y_test': y_test
        }
        print("")
    print("===========================================")
    best_encoding = max(results, key=lambda enc: results[enc]['accuracy'])
    print(f"Best performing encoding method: {best_encoding}")
    print(f"Test Accuracy: {results[best_encoding]['accuracy']:.4f}")
    
    compare_feature_importances(
        results[best_encoding]['model'],
        results[best_encoding]['X_test'],
        results[best_encoding]['y_test'],
        results[best_encoding]['features']
    )
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Execution time: {elapsed_time:4f}s")
    
if __name__ == '__main__':
    main()
