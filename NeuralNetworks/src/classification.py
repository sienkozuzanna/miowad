import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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

def visualize_predictions_classification(X_test, Y_test, Y_pred, num_classes):
   
    if Y_test.shape[1] > 1:
        Y_test_class = np.argmax(Y_test, axis=1)
    else:
        Y_test_class = Y_test

    plt.figure(figsize=(6, 4))
    
    for class_label in range(num_classes):
        correct_data = X_test[(Y_test_class == class_label) & (Y_pred == class_label)]
        plt.scatter(correct_data[:, 0], correct_data[:, 1], label=f"Correct Class {class_label}", alpha=0.6, marker='o')
        
        wrong_data = X_test[(Y_test_class == class_label) & (Y_pred != class_label)]
        plt.scatter(wrong_data[:, 0], wrong_data[:, 1], label=f"Wrong Class {class_label}", alpha=0.6, marker='x')

    plt.title("Prediction vs Actual Class", fontsize=14)
    plt.xlabel("x", fontsize=12)
    plt.ylabel("y", fontsize=12)
    plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), title="Classes")
    plt.grid(True)
    plt.show()
