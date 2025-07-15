import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error

def load_split_data(random_state=15):
    data = fetch_california_housing()
    X = data.data
    y = data.target
    feature_names = data.feature_names
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state)
    return X_train, X_test, y_train, y_test, feature_names

def linear_regression(X_train, y_train, X_test, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return model.intercept_, model.coef_, rmse

def ridge_regression(X_train, y_train, X_test, y_test):
    ridge = Ridge()
    param_grid = {'alpha': np.logspace(-15, 10, 10000)}
    grid = GridSearchCV(ridge, param_grid, scoring='neg_mean_squared_error')
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return grid.best_params_['alpha'], best_model.intercept_, best_model.coef_, rmse

def lasso_regression(X_train, y_train, X_test, y_test, max_iter=1000):
    lasso = Lasso(max_iter=max_iter)
    param_grid = {'alpha': np.logspace(-10, 10, 10000)}
    grid = GridSearchCV(lasso, param_grid, scoring='neg_mean_squared_error')
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return grid.best_params_['alpha'], best_model.intercept_, best_model.coef_, rmse

def lasso_features(coefs, feature_names):
    nonzero_idx = np.where(coefs != 0)[0]
    nonzero_feature = {}
    for idx in nonzero_idx:
        nonzero_feature[feature_names[idx]] = coefs[idx]
    return nonzero_feature

def main():
    X_train, X_test, y_train, y_test, feature_names = load_split_data(random_state=15)
    
    print("----- Linear Regression -----")
    intercept_lr, coef_lr, rmse_lr = linear_regression(X_train, y_train, X_test, y_test)
    print("Intercept:", intercept_lr)
    print("Coefficients:", coef_lr)
    print("RMSE:", rmse_lr)
    print()
    
    print("----- Ridge Regression -----")
    best_alpha_ridge, intercept_ridge, coef_ridge, rmse_ridge = ridge_regression(X_train, y_train, X_test, y_test)
    print("Best alpha:", best_alpha_ridge)
    print("Intercept:", intercept_ridge)
    print("Coefficients:", coef_ridge)
    print("RMSE:", rmse_ridge)
    print()
    
    print("----- Lasso Regression -----")
    best_alpha_lasso, intercept_lasso, coef_lasso, rmse_lasso = lasso_regression(X_train, y_train, X_test, y_test)
    print("Best alpha:", best_alpha_lasso)
    print("Intercept:", intercept_lasso)
    print("Coefficients:", coef_lasso)
    print("RMSE:", rmse_lasso)
    print()
    
    nonzero_features = lasso_features(coef_lasso, feature_names)
    print("Lasso에서 0이 아닌 계수를 가진 feature:")
    for feature, coef in nonzero_features.items():
        print(f"{feature}: {coef}")


if __name__ == "__main__":
    main()
