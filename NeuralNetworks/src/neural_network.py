import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score

class NeuralNetwork:
    def __init__(self, layers, activation, task="regression", last_layer_activation=None,weights_initialize=None, batch_size=None):
        """
        :param layers: A list specifying the number of neurons in each layer.
        :param activation: The activation function to use in the hidden layers.
        :param weights_initialize: Type of weight initialization (e.g., He, Xavier). If None -> random from uniform distribuiton U([0,1])
        :param task: Regression or classfication task
        :param last_layer_activation: Activation function for the last layer. If None:
                                  - 'linear' for regression
                                  - 'softmax' for classification
        """

        self.layers = layers
        self.batch_size = batch_size
        self.task=task

        if self.task=="regression":
            self.loss_function=self.MSE
            self.loss_function_name = "MSE"
        else:
            self.loss_function=self.CrossEntropy
            self.loss_function_name="CrossEntropy"

        if last_layer_activation is None:
            self.last_layer_activation = "linear" if self.task == "regression" else "softmax"
        else:
            self.last_layer_activation = last_layer_activation

        self.weights = []
        self.bias = []

        for i in range(len(layers)-1):
            if weights_initialize == "Xavier":
                std = np.sqrt(2 / (layers[i] + layers[i+1]))
                self.weights.append(np.random.randn(layers[i], layers[i+1]) * std)
            elif weights_initialize == "He":
                std = np.sqrt(2 / layers[i])
                self.weights.append(np.random.randn(layers[i], layers[i+1]) * std)
            else:
                self.weights.append(np.random.uniform(0, 1, size=(layers[i], layers[i+1])))

        self.bias = [np.random.uniform(-0.5, 0.5, size=(layers[i+1],)) for i in range(len(layers)-1)]

        activation_functions = {
            "sigmoid": self.sigmoid,
            "tanh": self.tanh,
            "relu": self.relu,
            "leaky_relu": self.leaky_relu,
            "linear": lambda x: x,
            "softmax": self.softmax
        }

        activation_functions_derivatives = {
            "sigmoid": self.sigmoid_derivative,
            "tanh": self.tanh_derivative,
            "relu": self.relu_derivative,
            "leaky_relu": self.leaky_relu_derivative,
            "linear": lambda x: 1,
            "softmax": None
        }

        self.activation_function = activation_functions.get(activation)
        self.activation_function_derivative=activation_functions_derivatives.get(activation)

        if last_layer_activation is None:
            if self.task == "regression":
                self.last_layer_activation = "linear"
            else:
                self.last_layer_activation = "softmax"

        self.last_layer_activation_function = activation_functions.get(self.last_layer_activation)
        self.last_layer_activation_function_derivative = activation_functions_derivatives.get(self.last_layer_activation)

        self.momentum_w = [np.zeros_like(w) for w in self.weights]
        self.momentum_b = [np.zeros_like(b) for b in self.bias]
        self.ema_w = [np.zeros_like(w) for w in self.weights]
        self.ema_b = [np.zeros_like(b) for b in self.bias]

    #activation functions
    def sigmoid(self, x): return 1 / (1 + np.exp(-x))
    def sigmoid_derivative(self, x): return self.sigmoid(x) * (1 - self.sigmoid(x))
    
    def tanh(self, x): return np.tanh(x)
    def tanh_derivative(self, x): return 1 - self.tanh(x)**2

    def relu(self, x): return np.maximum(0, x)
    def relu_derivative(self, x): return np.where(x > 0, 1, 0)
    
    def leaky_relu(self, x): return np.where(x > 0, x, 0.01 * x)
    def leaky_relu_derivative(self, x): return np.where(x > 0, 1, 0.01)

    def softmax(self, x):
        shift_x = x - np.max(x, axis=1, keepdims=True) 
        exp_x = np.exp(shift_x)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    #no need for softmax gradient since softmax+cross-entropy gradient = y_pred-y_true
    
    def forward(self, X):
        self.a = [X]
        self.z = []
        for i in range(len(self.weights)):
            z = np.dot(self.a[-1], self.weights[i]) + self.bias[i]
            self.z.append(z)
            if i == len(self.weights) - 1:
                a = self.last_layer_activation_function(z)
            else:
                a = self.activation_function(z)
            self.a.append(a)
        return self.a[-1]

    def compute_gradients(self, X, y, lambda_regularization=0.0, regularization_type="None"):
        m = X.shape[0]
        if self.task == "classification" and self.last_layer_activation == "softmax":
            delta = self.forward(X) - y  # softmax + cross-entropy gradient
        elif self.task == "regression":
            delta = (self.forward(X) - y) / m
        else:
            delta = (self.forward(X) - y) * self.last_layer_activation_function_derivative(self.z[-1])

        gradients_w, gradients_b = [], []
        total_weights = sum(w.size for w in self.weights)
        
        for i in reversed(range(len(self.weights))):
            gradient_w=np.dot(self.a[i].T, delta)
            if regularization_type == "L1":
                gradient_w += (lambda_regularization/total_weights) * np.sign(self.weights[i])
            elif regularization_type == "L2":
                gradient_w += (lambda_regularization/total_weights) * 2 * self.weights[i]
            gradients_w.append(gradient_w)
            gradients_b.append(np.sum(delta, axis=0))
            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * self.activation_function_derivative(self.z[i-1])
        return gradients_w[::-1], gradients_b[::-1]
    
    def update_weights(self, gradients_w, gradients_b, learning_rate):
        self.optimizer_function(gradients_w, gradients_b, learning_rate)

    def SGD_update(self, gradients_w, gradients_b, learning_rate):
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * gradients_w[i]
            self.bias[i] -= learning_rate * gradients_b[i]

    def momentum_update(self, gradients_w, gradients_b, learning_rate, momentum_coeff):
        for i in range(len(self.weights)):
            self.momentum_w[i] = gradients_w[i] + momentum_coeff * self.momentum_w[i]
            self.momentum_b[i] = gradients_b[i] + momentum_coeff * self.momentum_b[i]
            self.weights[i] -= learning_rate * self.momentum_w[i]
            self.bias[i] -= learning_rate * self.momentum_b[i]

    def RMSprop_update(self, gradients_w, gradients_b, learning_rate, beta, eps):
        for i in range(len(self.weights)):
            self.ema_w[i] = beta * self.ema_w[i] + (1 - beta) * (gradients_w[i] ** 2)
            self.ema_b[i] = beta * self.ema_b[i] + (1 - beta) * (gradients_b[i] ** 2)

            self.weights[i] -= learning_rate * gradients_w[i] / (np.sqrt(self.ema_w[i]) + eps)
            self.bias[i] -= learning_rate * gradients_b[i] / (np.sqrt(self.ema_b[i]) + eps)

    def set_optimizer(self, learning_rate, optimizer="GD", momentum_coeff=0.9, beta=0.9, eps=1e-8):
        self.optimizer = optimizer
        self.momentum_coeff = momentum_coeff
        self.beta = beta
        self.eps = eps

        if optimizer == "GD":
            self.optimizer_function = lambda gw, gb, lr: self.SGD_update(gw, gb, lr)
        elif optimizer == "momentum":
            self.optimizer_function = lambda gw, gb, lr: self.momentum_update(gw, gb, lr, momentum_coeff)
        elif optimizer == "RMSprop":
            self.optimizer_function = lambda gw, gb, lr: self.RMSprop_update(gw, gb, lr, beta, eps)
        else:
            raise ValueError("Unknown optimizer")

    def train(self, X_train, y_train, X_test, y_test, learning_rate, epochs, 
               optimizer="GD", momentum_coeff=0.9, beta=0.9, eps=1e-8, 
               batch_size=32, verbose=True, verbose_interval=100, plot_weights_update=False, weights_visualization_interval=1000,
               plot_training_loss=True, plot_test_loss=True, n_last_training_epochs=None, 
               return_training_stats=False, early_stopping=False, patience=10,
               lambda_regularization=0.00, regularization_type="None"):
        
        batch_size = batch_size if batch_size is not None else self.batch_size
        self.set_optimizer(learning_rate, optimizer, momentum_coeff, beta, eps)

        #storing train and test losses
        train_losses = []
        test_losses = []
        f1_scores = []

        best_test_loss=float('inf')
        epochs_without_improvement=0

        #saving best weights and biases
        best_weights = [w.copy() for w in self.weights]
        best_bias = [b.copy() for b in self.bias]

        start_time = time.time()

        for epoch in range(epochs):
            if batch_size is None: #full batch (all datasets records are used for computing gradient)
                gradients_w, gradients_b = self.compute_gradients(X_train, y_train, lambda_regularization, regularization_type)
                self.update_weights(gradients_w, gradients_b, learning_rate)
            else:
                permutation = np.random.permutation(X_train.shape[0]) #mini-batch
                for i in range(0, X_train.shape[0], batch_size):
                    indices = permutation[i:i+batch_size]
                    X_batch, y_batch = X_train[indices], y_train[indices]

                    gradients_w, gradients_b = self.compute_gradients(X_batch, y_batch, lambda_regularization, regularization_type)
                    self.update_weights(gradients_w, gradients_b, learning_rate)
            
            train_loss = self.loss_function(X_train, y_train, lambda_regularization, regularization_type)
            test_loss = self.loss_function(X_test, y_test, lambda_regularization, regularization_type)
            train_losses.append(train_loss)
            test_losses.append(test_loss)

            if self.task == "classification":
                f1_scores.append(self.Fscore(X_test, y_test))

            if verbose:
                if epoch % verbose_interval == 0:
                    print(f"Epoch {epoch}, Training Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}")

            if early_stopping:
                if test_loss<best_test_loss:
                    best_test_loss = test_loss
                    epochs_without_improvement=0
                    best_weights, best_bias= [w.copy() for w in self.weights], [b.copy() for b in self.bias]
                else:
                    epochs_without_improvement+=1
                
                if epochs_without_improvement>=patience:
                    print(f"Early stopping due to no improvment in test loss after {patience} epochs")
                    self.weights, self.bias = best_weights, best_bias
                    break

            if plot_weights_update:
                if (epoch+1) % weights_visualization_interval == 0:
                    self.plot_weights_distribution(epoch+1)

            
        
        end_time = time.time()
        #print(f"Total training time: {end_time - start_time:.2f} seconds")

        if plot_training_loss:
            self.plot_last_epochs_training(train_losses, test_losses, n_last_training_epochs, epochs,
                plot_test_loss,f1_scores=f1_scores if self.task == "classification" else None)

        if return_training_stats:
            return {
            "train_losses": train_losses,
            "test_losses": test_losses,
            "training_time": end_time - start_time,
        }
    
    def plot_last_epochs_training(self, train_losses, test_losses, n_last_training_epochs, n_total_epochs, plot_test_losses, f1_scores=None):
        actual_epochs_ran = len(train_losses)

        if n_last_training_epochs is not None:
            n_last_training_epochs = min(n_last_training_epochs, actual_epochs_ran)
            train_losses = train_losses[-n_last_training_epochs:]
            test_losses = test_losses[-n_last_training_epochs:]
            if f1_scores is not None:
                f1_scores = f1_scores[-n_last_training_epochs:]
            epoch_range = list(range(actual_epochs_ran - n_last_training_epochs, actual_epochs_ran))
        else:
            epoch_range = list(range(actual_epochs_ran))


        
        if f1_scores is None:
            plt.figure(figsize=(6, 4))
            plt.plot(epoch_range, train_losses, label="Training Loss", color="blue", linewidth=2)
            if plot_test_losses:
                plt.plot(epoch_range, test_losses, label="Test Loss", color="red", linewidth=2)
            plt.title("Losses During Training")
            plt.xlabel("Epoch")
            plt.ylabel(f"{self.loss_function_name}")
            plt.legend(loc='upper right')
            plt.grid(True)
            plt.show()
        else:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            ax1.plot(epoch_range, train_losses, label="Training Loss", color="blue", linewidth=2)
            if plot_test_losses:
                ax1.plot(epoch_range, test_losses, label="Test Loss", color="red", linewidth=2)
            ax1.set_title("Cross-Entropy Loss")
            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("Loss")
            ax1.legend()
            ax1.grid(True)

            ax2.plot(epoch_range, f1_scores, label="F1-score", color="blue", linewidth=2)
            ax2.set_title("F1-score over Epochs")
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("F1-score")
            ax2.legend()
            ax2.grid(True)

            plt.tight_layout()
            plt.show()
            plt.show()
    
    def plot_weights_heatmap(self, epoch):
        num_layers = len(self.weights)
        fig, axes = plt.subplots(1, num_layers, figsize=(4 * num_layers, 6))

        if num_layers == 1:
            axes = [axes]
        vmin = min(w.min() for w in self.weights)
        vmax = max(w.max() for w in self.weights)

        for i, (w, ax) in enumerate(zip(self.weights, axes)):
            sns.heatmap(w, annot=False, cmap="coolwarm", cbar=(i == num_layers - 1),
                        vmin=vmin, vmax=vmax, ax=ax,
                        cbar_ax=None if i < num_layers - 1 else fig.add_axes([0.92, 0.3, 0.02, 0.4]))
            ax.set_title(f"Weights Layer {i+1}")

        fig.suptitle(f"Weight Heatmaps at Epoch {epoch}", fontsize=16)
        plt.tight_layout(rect=[0, 0, 0.9, 0.95])
        plt.show()

    def plot_weights_distribution(self, epoch):
        plt.figure(figsize=(10, 5))
        
        for i, w in enumerate(self.weights):
            plt.subplot(1, len(self.weights), i + 1)
            sns.histplot(w.flatten(), kde=False, bins=30, color="royalblue")
            plt.title(f"Weight Distribution Layer {i+1}")
            plt.xlabel("Weight values")
            plt.ylabel("Frequency")

        plt.suptitle(f"Weight Distributions at Epoch {epoch}")
        plt.tight_layout()
        plt.show()

    def predict(self, X):
        if self.task=="regression":
            return self.forward(X)
        if self.task=="classification":
            return self.forward(X).argmax(axis=1)
    
    def MSE(self, X, Y, lambda_regularization=0.01, regularization_type="None"):
        mse = np.mean((self.predict(X) - Y) ** 2)
        total_weights = sum(w.size for w in self.weights)
        
        if regularization_type=="L1": 
            #L1 penetalny = lamda_regularization*sum(|w|)
            l1_penalty = lambda_regularization * (np.sum(np.abs(np.concatenate([w.flatten() for w in self.weights]))))/total_weights
            return mse + l1_penalty
        elif regularization_type=="L2":
            #L2 penetalny = lamda_regularization*sum(w^2)
            l2_penalty = lambda_regularization * (np.sum((np.concatenate([w.flatten()**2 for w in self.weights]))))/total_weights
            return mse + l2_penalty
        else:
            return mse
    
    def CrossEntropy(self, X, Y, lambda_regularization=0.01, regularization_type="None"):
        Y_pred = self.forward(X)
        epsilon = 1e-15
        Y_pred = np.clip(Y_pred, epsilon, 1 - epsilon) #to avoid log(0)
        cross_entropy = -np.mean(Y * np.log(Y_pred)) #for multi-class classification
        total_weights = sum(w.size for w in self.weights)
        if regularization_type=="L1": 
            #L1 penetalny = lamda_regularization*sum(|w|)
            l1_penalty = lambda_regularization * (np.sum(np.abs(np.concatenate([w.flatten() for w in self.weights]))))/total_weights
            return cross_entropy + l1_penalty
        elif regularization_type=="L2":
            #L2 penetalny = lamda_regularization*sum(w^2)
            l2_penalty = lambda_regularization * (np.sum((np.concatenate([w.flatten()**2 for w in self.weights]))))/total_weights
            return cross_entropy + l2_penalty
        else:
            return cross_entropy
        
    def Fscore(self, X, Y, average="micro"):
        Y_pred=self.predict(X)
        Y_true = np.argmax(Y, axis=1)

        f1 = f1_score(Y_true, Y_pred, average=average)
        return f1