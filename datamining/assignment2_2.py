import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def load_split_data(test_size=0.3, random_state=42):
    diabetes = load_diabetes()
    X, y = diabetes.data, diabetes.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    return X_train, X_test, y_train, y_test

def knn_model_fitting(X_train, y_train, X_test, y_test, k_range):
    results = []
    
    for scaling in [False, True]:
        if scaling:
            scaler = StandardScaler()
            # scaler = MinMaxScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
        else:
            X_train_scaled = X_train
            X_test_scaled = X_test
        
        cov = np.cov(X_train_scaled, rowvar=False)
        inv_cov = np.linalg.inv(cov)
        
        for metric in ['euclidean', 'mahalanobis']:
            for k in k_range:
                if metric == 'mahalanobis':
                    knn = KNeighborsRegressor(n_neighbors=k, 
                                              metric=metric, 
                                              metric_params={'V': inv_cov})
                else:
                    knn = KNeighborsRegressor(n_neighbors=k, metric=metric)
                    
                knn.fit(X_train_scaled, y_train)
                y_pred = knn.predict(X_test_scaled)
                mse = mean_squared_error(y_test, y_pred)
                
                results.append({
                    'scaling': scaling,
                    'metric': metric,
                    'k': k,
                    'mse': mse
                })
    
    df_results = pd.DataFrame(results)
    return df_results

def plot_knn_results(df_results):
    for scaling in [False, True]:
        plt.figure(figsize=(10, 6))
        for metric in ['euclidean', 'mahalanobis']:
            subset = df_results[(df_results['scaling'] == scaling) & (df_results['metric'] == metric)]
            plt.plot(subset['k'], subset['mse'], marker='o', label=metric)
        plt.xlabel("k")
        plt.ylabel("MSE")
        plt.title(f"scaling={scaling}")
        plt.legend()
        plt.show()

def main():
    X_train, X_test, y_train, y_test = load_split_data(test_size=0.3, random_state=42)
    df_results = knn_model_fitting(X_train, y_train, X_test, y_test, k_range=range(1, 11))
    print(df_results)
    plot_knn_results(df_results)

if __name__ == '__main__':
    main()
