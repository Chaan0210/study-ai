import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

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
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20],
        'min_samples_split': [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20],
        'min_samples_leaf': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    }
    tree = DecisionTreeClassifier(random_state=random_state)
    grid_search = GridSearchCV(
        tree, 
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

def plot_decision_tree(model, feature_names):
    plt.figure(figsize=(10, 8))
    plot_tree(
        model,
        feature_names=feature_names,
        class_names=['Died', 'Survived'],
        filled=True,
        rounded=True,
        fontsize=10
    )
    plt.title("Decision Tree")
    plt.tight_layout()
    plt.show()

def plot_feature_importances(model, feature_names):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    sorted_features = np.array(feature_names)[indices]
    sorted_importances = importances[indices]
    
    plt.figure(figsize=(10, 8))
    plt.title("Feature Importances")
    plt.bar(range(len(sorted_features)), sorted_importances, align='center')
    plt.xticks(range(len(sorted_features)), sorted_features, rotation=45, ha='right')
    plt.ylabel("Importance")
    plt.xlabel("Features")
    plt.tight_layout()
    plt.show()


def main():
    file_path = './titanic.csv'
    encoding_methods = ['label', 'onehot']
    results = {}

    for encoding in encoding_methods:
        print(f"----- Encoding Method: {encoding} -----")
        X_train, y_train, X_test, y_test = load_preprocess_data(file_path, test_size=0.2, random_state=1, encoding=encoding)
        best_tree = fit_model(X_train, y_train, random_state=1)
        test_acc = evaluate_model(best_tree, X_test, y_test)
        results[encoding] = {
            'model': best_tree,
            'accuracy': test_acc,
            'features': X_train.columns
        }
    print("===========================================")
    print("")
    best_encoding = max(results, key=lambda enc: results[enc]['accuracy'])
    print(f"Best performing encoding method: {best_encoding}")
    print(f"Test Accuracy: {results[best_encoding]['accuracy']:.4f}")
    
    plot_decision_tree(results[best_encoding]['model'], results[best_encoding]['features'])
    plot_feature_importances(results[best_encoding]['model'], results[best_encoding]['features'])
    

if __name__ == '__main__':
    main()
