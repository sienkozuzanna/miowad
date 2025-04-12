import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def one_hot_encoding(Y, num_classes):
    one_hot = np.zeros((len(Y), num_classes))
    one_hot[np.arange(len(Y)), Y.flatten().astype(int)] = 1 
    return one_hot

def load_data_classification(training_path, test_path, index_col=None, encoding=True):
    training = pd.read_csv(training_path, index_col=index_col)
    test = pd.read_csv(test_path, index_col=index_col)

    X_train, Y_train = training[['x', 'y']].to_numpy(), training['c'].to_numpy().reshape(-1, 1)
    X_test, Y_test = test[['x', 'y']].to_numpy(), test['c'].to_numpy().reshape(-1, 1)
    num_classes = len(np.unique(Y_train))

    if encoding:
        Y_train = one_hot_encoding(Y_train, num_classes)
        Y_test = one_hot_encoding(Y_test, num_classes)

    return X_train, Y_train, X_test, Y_test, num_classes

def visualize_data_classification(X_train, Y_train, num_classes):
    if Y_train.shape[1] > 1:
        Y_train_class = np.argmax(Y_train, axis=1)

    plt.figure(figsize=(6, 4))
    for class_label in range(num_classes):
        class_data = X_train[Y_train_class == class_label]
        plt.scatter(class_data[:, 0], class_data[:, 1], label=f"Class {class_label}", alpha=0.6)

    plt.title("Visualization of Training Data", fontsize=14)
    plt.xlabel("x", fontsize=12)
    plt.ylabel("y", fontsize=12)
    plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), title="Classes")
    plt.grid(True)
    plt.show()


def visualize_predictions_classification(model, X_test, Y_test, Y_pred, num_classes):
    colors = sns.color_palette("tab10", num_classes)

    if Y_test.shape[1] > 1:
        Y_test_class = np.argmax(Y_test, axis=1)
    else:
        Y_test_class = Y_test.flatten()

    Y_pred_class = np.argmax(Y_pred, axis=1) if Y_pred.ndim > 1 else Y_pred.flatten()

    plt.figure(figsize=(6, 4))

    if model is not None:
        x_min, x_max = X_test[:, 0].min() - 1, X_test[:, 0].max() + 1
        y_min, y_max = X_test[:, 1].min() - 1, X_test[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                             np.linspace(y_min, y_max, 300))
        grid = np.c_[xx.ravel(), yy.ravel()]
        Z = model.predict(grid)
        Z = np.argmax(Z, axis=1) if Z.ndim > 1 else Z
        Z = Z.reshape(xx.shape)

        plt.contourf(xx, yy, Z, alpha=0.2, levels=np.arange(num_classes + 1) - 0.5,
                     colors=colors, linestyles='dotted')

    for class_label in range(num_classes):
        correct_data = X_test[(Y_test_class == class_label) & (Y_pred_class == class_label)]
        wrong_data = X_test[(Y_test_class == class_label) & (Y_pred_class != class_label)]

        plt.scatter(correct_data[:, 0], correct_data[:, 1],
                    color=colors[class_label], alpha=0.6, marker='o', label=f"Correct: {class_label}")
        plt.scatter(wrong_data[:, 0], wrong_data[:, 1],
                    color='black', alpha=0.4, marker='x', label=f"Wrong: {class_label}")

    plt.title("Prediction vs Actual Class", fontsize=14)
    plt.xlabel("x", fontsize=12)
    plt.ylabel("y", fontsize=12)
    plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), title="Predictions")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
