import pandas as pd
import matplotlib.pyplot as plt

def load_data(training_path, test_path, index_col=0):
    training=pd.read_csv(training_path, index_col=index_col)
    test=pd.read_csv(test_path, index_col=index_col)

    X_train, Y_train=training['x'].to_numpy().reshape(-1,1), training['y'].to_numpy().reshape(-1,1)
    X_test, Y_test=test['x'].to_numpy().reshape(-1,1), test['y'].to_numpy().reshape(-1,1)

    return X_train, Y_train, X_test, Y_test

def plot_fitted_vs_actual(X_train, Y_train, Y_pred, train_or_test):
    plt.figure(figsize=(6, 4))
    plt.scatter(X_train, Y_pred, color='r', label="Predicted values")
    plt.scatter(X_train, Y_train, color='b', label="True values")
    plt.xlabel(f'X_{train_or_test}')
    plt.ylabel('Y')
    plt.legend(loc='lower right')
    plt.title(f'Fitted vs Actual values for {train_or_test} data')
    plt.grid(True)
    plt.show()