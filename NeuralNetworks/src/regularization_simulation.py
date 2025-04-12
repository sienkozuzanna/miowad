from neural_network import NeuralNetwork
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from IPython.display import display
from matplotlib.lines import Line2D

def train_architecture_with_regularization(X_train, Y_train, X_test, Y_test, 
                              task, layers, activation_function, epochs, batch_size, 
                              learning_rate, optimizer, 
                              regularization_types, regularization_values, data_normalized=False, normalizer=None):
    results = {
        'regularization': [],
        'regularization_values': [],
        'models': [],
        'training_times': [],
        'train_losses': [],
        'test_losses': [],
        'eval_metrics': [],
        'Y_pred': []
    }
    if activation_function in ['relu', 'leaky_relu']:
            weights_initialize = 'He'
    else:
        weights_initialize = 'Xavier'
    
    for i, regularization in enumerate(regularization_types):
        print(f'Training with regularization: {regularization}')
        
        values = regularization_values[i] if isinstance(regularization_values[i], list) else [regularization_values[i]]
        
        for val in values:
            print(f'  λ = {val}')
            nn = NeuralNetwork(layers=layers, task=task, activation=activation_function, weights_initialize=weights_initialize)
            training_stats = nn.train(X_train, Y_train, X_test, Y_test, learning_rate=learning_rate,
                                    batch_size=batch_size, epochs=epochs, verbose=False, plot_training_loss=False,
                                    optimizer=optimizer, return_training_stats=True,
                                    regularization_type=regularization, lambda_regularization=val)

            if task == "classification":
                y_pred = nn.predict(X_test)
                eval_metric = nn.Fscore(X_test, Y_test)
            else:
                if data_normalized:
                    y_pred=normalizer.denormalize_Y(nn.predict(X_test))
                    y_test_denorm = normalizer.denormalize_Y(Y_test)
                    eval_metric = np.mean((y_pred - y_test_denorm) ** 2)
                else:
                    y_pred = nn.predict(X_test)
                    eval_metric = nn.MSE(X_test, Y_test)

            results['regularization'].append(regularization)
            results['regularization_values'].append(val)
            results['train_losses'].append(training_stats['train_losses'])
            results['test_losses'].append(training_stats['test_losses'])
            results['training_times'].append(training_stats['training_time'])
            results['eval_metrics'].append(eval_metric)
            results['Y_pred'].append(y_pred)
            results['models'].append(nn)

    return results

def plot_losses(results, log_scale=False, plot_interval=100):
    regularization = results['regularization']
    regularization_values = results['regularization_values']
    train_losses = results['train_losses']
    test_losses = results['test_losses']
    plt.style.use('default')
    plt.figure(figsize=(12, 8)) 
    
    colors = sns.color_palette("tab10", len(regularization_values))
    linewidth = 1.2
    train_linestyle = '-'
    test_linestyle = ':'
    
    for i, reg in enumerate(regularization):
        lambda_val = f"{regularization_values[i]:g}" 
        train_epochs = list(range(0, len(train_losses[i]), plot_interval))
        test_epochs = list(range(0, len(test_losses[i]), plot_interval))
        
        plt.plot(train_epochs, [train_losses[i][epoch] for epoch in train_epochs], 
                 label=f'Train - λ={lambda_val}', color=colors[i], linestyle=train_linestyle, linewidth=linewidth)
        
        plt.plot(test_epochs, [test_losses[i][epoch] for epoch in test_epochs], 
                 label=f'Test - λ={lambda_val}', color=colors[i], linestyle=test_linestyle, linewidth=linewidth)

    
    plt.xlabel('Epochs', fontsize=14, labelpad=10)
    plt.ylabel('Loss', fontsize=14, labelpad=10)
    if log_scale:
        plt.yscale('log')
        plt.ylabel('Loss (log scale)', fontsize=14, labelpad=10)
    plt.title(f'Training and Test Losses with {regularization[0]} Regularization', fontsize=16, pad=20)
    

    legend = plt.legend(bbox_to_anchor=(1.05, 1), 
                       loc='upper left', 
                       fontsize=12,
                       title='Regularization Type\n(λ values)',
                       title_fontsize=13,
                       frameon=True,
                       shadow=True)
    
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.xlim(0, len(train_losses[0])-1)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    
    plt.show()


def plot_fitted_vs_actual_regularization_regression(results, X_test, Y_test):
    regularization = results['regularization']
    regularization_values = results['regularization_values']
    Y_preds = results['Y_pred']
    
    num_plots = len(regularization)
    num_cols = min(2, num_plots)
    num_rows = int(np.ceil(num_plots / num_cols))
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 4 * num_rows))
    axes = axes.flatten() if num_plots > 1 else [axes]
    
    for i, (reg, val, Y_pred) in enumerate(zip(regularization, regularization_values, Y_preds)):
        ax = axes[i]
        
        X_flat = X_test.flatten() if X_test.ndim > 1 else X_test
        Y_pred_flat = Y_pred.flatten() if Y_pred.ndim > 1 else Y_pred
        Y_test_flat = Y_test.flatten() if Y_test.ndim > 1 else Y_test
        
        sns.scatterplot(x=X_flat, y=Y_pred_flat, color='r', ax=ax, label='Predicted')
        sns.scatterplot(x=X_flat, y=Y_test_flat, color='b', ax=ax, label='True')

        ax.set_title(f'{reg} (λ={val})')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.grid(True)
        ax.legend(loc='lower right', fontsize=10)

    
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='b', markersize=7, label='True values'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='r', markersize=7, label='Predicted values')
    ]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.12, 0.9))
    
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.show()


def plot_classification_results(results, X_test, Y_test, num_classes):
    regs = results['regularization']
    reg_vals = results['regularization_values']
    y_preds = results['Y_pred']
    models = results['models']
    
    labels = [f"{r} (λ={v})" for r, v in zip(regs, reg_vals)]

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
        
        ax.set_title(f'{label}')
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

def visualize_regularization_results(results, task, X_test, Y_test, num_classes=None, log_scale=False, data_normalized=False, normalizer=None):
    regs = results['regularization']
    reg_vals = results['regularization_values']
    training_times = results['training_times']
    eval_metrics = results['eval_metrics']

    labels = [f"{r} (λ={v})" for r, v in zip(regs, reg_vals)]

    plot_losses(results, log_scale=log_scale)
    if task == "regression":
            plot_fitted_vs_actual_regularization_regression(results, X_test, Y_test)
    elif task == "classification":
        plot_classification_results(results, X_test, Y_test, num_classes)

    metric_name = "MSE" if task == "regression" else "Fscore"
    eval_metrics_rounded = pd.Series(np.round(eval_metrics, 5)).apply(
        lambda x: f"{x:.2f}".rstrip('0').rstrip('.') if '.' in f"{x:.2f}" else f"{x}"
    )
    df_eval = pd.DataFrame({
        'Regularization': labels,
        f'{task.capitalize()} Eval Metric ({metric_name})': eval_metrics_rounded
    }).style.set_properties(**{'width': '350px', 'text-align': 'left'})
    
    print(f"{task.capitalize()} Task Evaluation Metric per Regularization")
    display(df_eval)

    training_times_rounded = pd.Series(np.round(training_times, 5)).apply(
        lambda x: f"{x:.2f}".rstrip('0').rstrip('.') if '.' in f"{x:.2f}" else f"{x}"
    )
    df_time = pd.DataFrame({
        'Regularization': labels,
        'Training Time (s)': training_times_rounded
    }).style.set_properties(**{'width': '350px', 'text-align': 'left'})
    
    print("Training Time per Regularization")
    display(df_time)

