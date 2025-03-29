import importlib
import sys
import os
sys.path.append(os.path.abspath("src"))
import neural_network
importlib.reload(neural_network)
from neural_network import NeuralNetwork
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def train_architecture_with_activation_functions(task, layers, activations, epochs, batch_size, 
                                                 learning_rate, optimizer,
                                                 X_train, Y_train, X_test, Y_test, data_normalized=False, normalizer=None):
    results = {
        'activations': [],
        'training_times': [],
        'train_losses': [],
        'test_losses': [],
        'eval_metrics': [],
        'Y_pred': []
    }

    for activation in activations:
        print(f'Training architecture with {activation}')
        if activation in ['relu', 'leaky_relu']:
            weights_initialize = 'He'
        else:
            weights_initialize = 'Xavier'
        nn=NeuralNetwork(layers=layers, task=task,
                         activation=activation, weights_initialize=weights_initialize)
        training_stats=nn.train(X_train, Y_train, X_test, Y_test, learning_rate=learning_rate,
                          batch_size=batch_size, epochs=epochs, verbose=False, plot_training_loss=False,
                          optimizer=optimizer, return_training_stats=True)
        if data_normalized==True:
            y_pred = nn.predict(X_test)
            y_pred_denorm = normalizer.denormalize_Y(y_pred)
            y_test_denorm = normalizer.denormalize_Y(Y_test)
            eval_metric = np.mean((y_pred_denorm - y_test_denorm) ** 2)
            results['Y_pred'].append(y_pred_denorm)
            
        else:
            y_pred = nn.predict(X_test)
            results['Y_pred'].append(y_pred)
            if task=="classification":
                eval_metric=nn.Fscore(X_test, Y_test)
            else:
                eval_metric=nn.MSE(X_test, Y_test)

        results['activations'].append(activation)
        results['train_losses'].append(training_stats['train_losses'])
        results['test_losses'].append(training_stats['test_losses'])
        results['training_times'].append(training_stats['training_time'])
        results['eval_metrics'].append(eval_metric)

    return results

def plot_losses(activations, train_losses, test_losses):
    plt.figure(figsize=(10,6))
    colors = sns.color_palette("tab10", len(activations))

    for i, activation in enumerate(activations):
        color = colors[i]
        plt.plot(train_losses[i], label=f'Train Loss - {activation}', color=color)
        plt.plot(test_losses[i], label=f'Test Loss - {activation}', linestyle='--', color=color)

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Test Losses per Activation Function')
    
    plt.legend(loc='upper left', bbox_to_anchor=(1.1, 0.9), title="Activation Functions")
    plt.grid(True)
    plt.show()

def display_eval_metrics(activations, eval_metrics, task):
    if task=="regression":
        metric="MSE"
    else:
        metric="Fscore"
    eval_metrics_rounded = pd.Series(np.round(eval_metrics, 2)).apply(lambda x: f"{x:.2f}".rstrip('0').rstrip('.'))
    metrics_df = pd.DataFrame({
        'Activation Function': activations,
        f'{task.capitalize()} Task Eval Metric ({metric})': eval_metrics_rounded
    })
    metrics_df = metrics_df.style.set_properties(**{
        'width': '250px', 
        'text-align': 'left'
    })
    
    print(f"{task.capitalize()} Task Evaluation Metric per Activation Function")
    display(metrics_df)
    return metrics_df

def display_training_times(activations, training_times):
    training_times_rounded=pd.Series(np.round(training_times,2)).apply(lambda x: f"{x:.2f}".rstrip('0').rstrip('.'))
    training_times_df = pd.DataFrame({
        'Activation Function': activations,
        'Training Time (seconds)': training_times_rounded
    })

    training_times_df = training_times_df.style.set_properties(**{
        'width': '250px', 
        'text-align': 'left'
    })
    print("Training Time per Activation Function")
    display(training_times_df)
    return training_times_df

def plot_fitted_vs_actual_for_activations(activations, X_test, Y_test, Y_preds):
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    axes = axes.flatten()
   
    for i, activation in enumerate(activations):
        ax = axes[i]
        Y_pred = Y_preds[i]

        X_flat = X_test.flatten() if X_test.ndim > 1 else X_test
        Y_pred_flat = Y_pred.flatten() if Y_pred.ndim > 1 else Y_pred
        Y_test_flat = Y_test.flatten() if Y_test.ndim > 1 else Y_test
    
        sns.scatterplot(x=X_flat, y=Y_pred_flat, color='r', ax=ax, label='_nolegend_')
        sns.scatterplot(x=X_flat, y=Y_test_flat, color='b', ax=ax, label='_nolegend_')
        
        ax.set_title(f'{activation} Activation')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.grid(True)
    
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='b', markersize=7, label='True values'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='r', markersize=7, label='Predicted values')
    ]
    
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.20, 0.9))
    
    plt.tight_layout()
    plt.show()

def plot_fitted_vs_actual_for_classification(activations, X_test, Y_test, Y_preds, num_classes):
    num_rows = int(np.ceil(len(activations) / 2))
    fig, axes = plt.subplots(num_rows, 2, figsize=(10, 3 * num_rows))
    axes = axes.flatten()
    
    if Y_test.shape[1] > 1:
        Y_test_class = np.argmax(Y_test, axis=1)
    else:
        Y_test_class = Y_test.flatten()
    
    colors = sns.color_palette("tab10", num_classes)
    
    for i, activation in enumerate(activations):
        ax = axes[i]
        Y_pred_class = np.argmax(Y_preds[i], axis=1) if Y_preds[i].ndim > 1 else Y_preds[i].flatten()
        
        for class_label in range(num_classes):
            correct_data = X_test[(Y_test_class == class_label) & (Y_pred_class == class_label)]
            wrong_data = X_test[(Y_test_class == class_label) & (Y_pred_class != class_label)]
            
            ax.scatter(correct_data[:, 0], correct_data[:, 1], color=colors[class_label], alpha=0.6, marker='o')
            ax.scatter(wrong_data[:, 0], wrong_data[:, 1], color=colors[class_label], alpha=0.6, marker='x')
        
        ax.set_title(f'{activation} Activation')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.grid(True)
    
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.show()


def visualize_results(results, task, X_test, Y_test, num_classes=None):
    activations = results['activations']
    training_times = results['training_times']
    train_losses = results['train_losses']
    test_losses = results['test_losses']
    eval_metrics = results['eval_metrics']
    y_pred=results['Y_pred']
    
    plot_losses(activations, train_losses, test_losses)
    if task=="regression":
        plot_fitted_vs_actual_for_activations(activations=activations, X_test=X_test, Y_test=Y_test, Y_preds=y_pred)
    else:
        plot_fitted_vs_actual_for_classification(activations=activations, X_test=X_test, Y_test=Y_test, Y_preds=y_pred, num_classes=num_classes)
    display_eval_metrics(activations, eval_metrics, task=task)
    display_training_times(activations, training_times)
