import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

def load_split_data(test_size=0.3, random_state=42, stratify=False):
    cancer = load_breast_cancer()
    X, y = cancer.data, cancer.target
    stratify_param = y if stratify else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify_param)

    return X_train, X_test, y_train, y_test

def knn_model_fitting(X_train, y_train, X_test, y_test, k_list, p_list):
    best_accuracy = 0
    best_params = []
    results = []

    for p in p_list:
        accuracy_list = []
        for k in k_list:
            knn = KNeighborsClassifier(n_neighbors=k, p=p)
            knn.fit(X_train, y_train)
            y_pred = knn.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            accuracy_list.append(accuracy)
            f1 = f1_score(y_test, y_pred)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params = [(k, p, f1)]
            elif accuracy == best_accuracy:
                best_params.append((k, p, f1))
        results.append((p, accuracy_list))

    return results, best_accuracy, best_params

def plot_results(results, k_values):
    plt.figure(figsize=(10, 6))
    for p, acc_list in results:
        plt.plot(list(k_values), acc_list, marker='o', label=f'p={p}')
    plt.xlabel("k")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    k_list = range(1, 31)
    p_list = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10]

    X_train, X_test, y_train, y_test = load_split_data(test_size=0.3, random_state=42, stratify=False)
    
    results, best_accuracy, best_params = knn_model_fitting(X_train, y_train, X_test, y_test, k_list, p_list)

    print("Stratify = False")
    plot_results(results, k_list)
    
    print(f"Best Accuracy: {best_accuracy:.4f}")
    print("Best (k, p):", end=" ")
    for k, p, f1 in best_params:
        print(f"({k}, {p}) - f1 score: {f1:.4f}", end=" ")

    X_train, X_test, y_train, y_test = load_split_data(test_size=0.3, random_state=42, stratify=True)
    
    results, best_accuracy, best_params = knn_model_fitting(X_train, y_train, X_test, y_test, k_list, p_list)

    print("\n\nStratify = True")
    plot_results(results, k_list)

    print(f"Best Accuracy: {best_accuracy:.4f}")
    print("Best (k, p):", end=" ")
    for k, p, f1 in best_params:
        print(f"({k}, {p}) - f1 score: {f1:.4f}", end=" ")

if __name__ == '__main__':
    main()
