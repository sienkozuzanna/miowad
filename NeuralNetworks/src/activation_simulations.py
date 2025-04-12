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
        'Y_pred': [], 
        'models': []
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
        results['models'].append(nn)

    return results

def plot_losses(activations, train_losses, test_losses, log_scale=False):
    plt.figure(figsize=(10,6))
    colors = sns.color_palette("tab10", len(activations))

    for i, activation in enumerate(activations):
        color = colors[i]
        plt.plot(train_losses[i], label=f'Train Loss - {activation}', color=color, linewidth=0.7)
        plt.plot(test_losses[i], label=f'Test Loss - {activation}', linestyle=':', color=color, linewidth=0.7)

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    if log_scale:
        plt.yscale('log')
        plt.title('Training and Test Losses per Activation Function (Log Scale)')
    else:
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

def plot_classification_results(results, X_test, Y_test, num_classes):
    activations = results['activations']  # list of activation function names
    y_preds = results['Y_pred']           # list of predictions for each activation
    models = results['models']            # list of models for each activation
    
    labels = [f"Activation: {act}" for act in activations]
    
    num_plots = len(labels)
    num_cols = min(2, num_plots)
    num_rows = int(np.ceil(num_plots / num_cols))
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 4 * num_rows))
    
    if num_plots == 1:
        axes = np.array([axes])
    
    axes = axes.flatten()
    
    if Y_test.shape[1] > 1:
        Y_test_class = np.argmax(Y_test, axis=1)
    else:
        Y_test_class = Y_test.flatten()
    
    colors = sns.color_palette("tab10", num_classes)
    
    all_handles = []
    all_labels = []
    
    for class_label in range(num_classes):
        correct_patch = plt.Line2D([], [], color=colors[class_label], marker='o', linestyle='None',
                                   markersize=8, label=f'Class {class_label} (correct)')
        wrong_patch = plt.Line2D([], [], color=colors[class_label], marker='x', linestyle='None',
                                 markersize=8, label=f'Class {class_label} (wrong)')
        all_handles.extend([correct_patch, wrong_patch])
        all_labels.extend([f'Class {class_label} (correct)', f'Class {class_label} (wrong)'])
    
    for i, (label, model) in enumerate(zip(labels, models)):
        ax = axes[i]
        Y_pred_class = np.argmax(y_preds[i], axis=1) if y_preds[i].ndim > 1 else y_preds[i].flatten()
        
        x_min, x_max = X_test[:, 0].min() - 0.5, X_test[:, 0].max() + 0.5
        y_min, y_max = X_test[:, 1].min() - 0.5, X_test[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                             np.linspace(y_min, y_max, 300))
        grid = np.c_[xx.ravel(), yy.ravel()]
        
        Z = model.predict(grid)
        Z = np.argmax(Z, axis=1) if Z.ndim > 1 else Z
        Z = Z.reshape(xx.shape)
        
        ax.contourf(xx, yy, Z, alpha=0.2, levels=np.arange(num_classes + 1) - 0.5,
                   colors=colors, linestyles='dotted')
        
        for class_label in range(num_classes):
            correct_data = X_test[(Y_test_class == class_label) & (Y_pred_class == class_label)]
            wrong_data = X_test[(Y_test_class == class_label) & (Y_pred_class != class_label)]
            
            ax.scatter(correct_data[:, 0], correct_data[:, 1], color=colors[class_label], 
                      alpha=0.6, marker='o')
            ax.scatter(wrong_data[:, 0], wrong_data[:, 1], color=colors[class_label], 
                      alpha=0.6, marker='x')
        
        ax.set_title(label)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.grid(True)
    
    fig.legend(handles=all_handles, labels=all_labels,
               loc='upper right', bbox_to_anchor=(1.05, 0.9),
               ncol=1, fontsize=9)
    
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.show()


def visualize_results(results, task, X_test, Y_test, num_classes=None, log_scale=False):
    activations = results['activations']
    training_times = results['training_times']
    train_losses = results['train_losses']
    test_losses = results['test_losses']
    eval_metrics = results['eval_metrics']
    y_pred=results['Y_pred']
    model=results['models']
    
    plot_losses(activations, train_losses, test_losses, log_scale)
    if task=="regression":
        plot_fitted_vs_actual_for_activations(activations=activations, X_test=X_test, Y_test=Y_test, Y_preds=y_pred)
    else:
        plot_classification_results(results, X_test, Y_test, num_classes)
    display_eval_metrics(activations, eval_metrics, task=task)
    display_training_times(activations, training_times)
