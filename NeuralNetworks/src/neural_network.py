import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns

class NeuralNetwork:
    def __init__(self, layers, activation, learning_rate, weights_initialize=None, batch_size=None):
        """
        :param layers: A list specifying the number of neurons in each layer.
        :param activation: The activation function to use in the hidden layers.
        :param weights_initialize: Type of weight initialization (e.g., He, Xavier). If None -> random from uniform distribuiton U([0,1])
        """

        self.layers = layers
        self.weights = []
        self.bias = []
        self.learning_rate = learning_rate
        self.batch_size = batch_size

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
            "leaky_relu": self.leaky_relu
        }

        activation_functions_derivatives = {
            "sigmoid": self.sigmoid_derivative,
            "tanh": self.tanh_derivative,
            "relu": self.relu_derivative,
            "leaky_relu": self.leaky_relu_derivative
        }

        self.activation_function = activation_functions.get(activation)
        self.activation_function_derivative=activation_functions_derivatives.get(activation)

        self.momentum_w = [np.zeros_like(w) for w in self.weights]
        self.momentum_b = [np.zeros_like(b) for b in self.bias]
        self.ema_w = [np.zeros_like(w) for w in self.weights]
        self.ema_b = [np.zeros_like(b) for b in self.bias]

    #activation functions
    def sigmoid(self, x): return 1 / (1 + np.exp(-x))
    def sigmoid_derivative(self, x): return x * (1 - x)
    
    def tanh(self, x): return np.tanh(x)
    def tanh_derivative(self, x): return 1-x**2
    
    def relu(self, x): return np.maximum(0, x)
    def relu_derivative(self, x): return np.where(x > 0, 1, 0)
    
    def leaky_relu(self, x): return np.where(x > 0, x, 0.01 * x)
    def leaky_relu_derivative(self, x): return np.where(x > 0, 1, 0.01)

    
    def forward(self, X):
        self.a = [X]
        self.z = []
        for i in range(len(self.weights)-1):
            z = np.dot(self.a[-1], self.weights[i]) + self.bias[i]
            self.z.append(z)
            a = self.activation_function(z)
            self.a.append(a)

        z = np.dot(self.a[-1], self.weights[-1]) + self.bias[-1]
        self.z.append(z)
        self.a.append(z)
        
        return self.a[-1]

    def compute_gradients(self, X, y):
        m = X.shape[0]
        delta = (self.forward(X) - y)/m 
        gradients_w, gradients_b = [], []
        
        for i in reversed(range(len(self.weights))):
            gradients_w.append(np.dot(self.a[i].T, delta))
            gradients_b.append(np.sum(delta, axis=0))
            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * self.activation_function_derivative(self.z[i-1])
        return gradients_w[::-1], gradients_b[::-1]
    
    def update_weights(self, gradients_w, gradients_b):
        self.optimizer_function(gradients_w, gradients_b)

    def SGD_update(self, gradients_w, gradients_b):
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * gradients_w[i]
            self.bias[i] -= self.learning_rate * gradients_b[i]

    def momentum_update(self, gradients_w, gradients_b, momentum_coeff):
        for i in range(len(self.weights)):
            self.momentum_w[i] = gradients_w[i] + momentum_coeff * self.momentum_w[i]
            self.momentum_b[i] = gradients_b[i] + momentum_coeff * self.momentum_b[i]
            self.weights[i] -= self.learning_rate * self.momentum_w[i]
            self.bias[i] -= self.learning_rate * self.momentum_b[i]

    def RMSprop_update(self, gradients_w, gradients_b, beta, eps):
        for i in range(len(self.weights)):
            self.ema_w[i] = beta * self.ema_w[i] + (1 - beta) * (gradients_w[i] ** 2)
            self.ema_b[i] = beta * self.ema_b[i] + (1 - beta) * (gradients_b[i] ** 2)

            self.weights[i] -= self.learning_rate * gradients_w[i] / (np.sqrt(self.ema_w[i]) + eps)
            self.bias[i] -= self.learning_rate * gradients_b[i] / (np.sqrt(self.ema_b[i]) + eps)

    def set_optimizer(self, optimizer="SGD", momentum_coeff=0.9, beta=0.9, eps=1e-8):
        self.optimizer = optimizer
        self.momentum_coeff = momentum_coeff
        self.beta = beta
        self.eps = eps

        if optimizer == "SGD":
            self.optimizer_function = self.SGD_update
        elif optimizer == "momentum":
            self.optimizer_function = lambda gw, gb: self.momentum_update(gw, gb, momentum_coeff)
        elif optimizer == "RMSprop":
            self.optimizer_function = lambda gw, gb: self.RMSprop_update(gw, gb, beta, eps)
        else:
            raise ValueError("Unknown optimizer")

    def train(self, X_train, y_train, X_test, y_test, epochs, 
               optimizer="SGD", momentum_coeff=0.9, beta=0.9, eps=1e-8, 
               batch_size=32, verbose=True, verbose_interval=100, 
               plot_weights_upadate=False, weights_visualization_interval=1000,
               plot_training_loss=True, n_last_training_epochs=None):
        
        batch_size = batch_size if batch_size is not None else self.batch_size
        self.set_optimizer(optimizer, momentum_coeff, beta, eps)
        train_losses = []
        test_losses = []
        start_time = time.time()

        for epoch in range(epochs):
            if batch_size is None: #full batch (all datasets records are used for computing gradient)
                gradients_w, gradients_b = self.compute_gradients(X_train, y_train)
                self.update_weights(gradients_w, gradients_b)
            else:
                permutation = np.random.permutation(X_train.shape[0]) #mini-batch
                for i in range(0, X_train.shape[0], batch_size):
                    indices = permutation[i:i+batch_size]
                    X_batch, y_batch = X_train[indices], y_train[indices]

                    gradients_w, gradients_b = self.compute_gradients(X_batch, y_batch)
                    self.update_weights(gradients_w, gradients_b)
            
            train_loss = self.MSE(X_train, y_train)
            test_loss = self.MSE(X_test, y_test)
            train_losses.append(train_loss)
            test_losses.append(test_loss)

            if verbose:
                if epoch % 100 == 0:
                    print(f"Epoch {epoch}, Training Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}")
        
            if plot_weights_upadate:
                if (epoch+1) % weights_visualization_interval == 0:
                    self.plot_weights_distribution(epoch+1)
            
        
        end_time = time.time()
        print(f"Total training time: {end_time - start_time:.2f} seconds")

        if plot_training_loss:
            self.plot_last_epochs_training(self.train_losses, n_last_training_epochs, epochs)
    
    def plot_last_epochs_training(self, train_losses, n_last_training_epochs, n_total_epochs):

        if n_last_training_epochs is not None:
            train_losses=train_losses[-n_last_training_epochs:]
            epoch_range = list(range(n_total_epochs - n_last_training_epochs, n_total_epochs))
        else:
            epoch_range=list(range(n_total_epochs))

        plt.figure(figsize=(6, 4))
        plt.plot(epoch_range, train_losses, label="Training Loss", color="blue", linewidth=2)
        plt.title("Training Losses")
        plt.xlabel("Epoch")
        plt.ylabel("MSE")
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def plot_weights_heatmap(self, epoch):
        plt.figure(figsize=(6, 4))
        for i, w in enumerate(self.weights):
            plt.subplot(1, len(self.weights), i + 1)
            sns.heatmap(w, annot=False, cmap="Blues", cbar=True)
            plt.title(f"Weights Layer {i+1} at Epoch {epoch}")
        plt.suptitle(f"Weight Heatmaps at Epoch {epoch}")
        plt.tight_layout()
        plt.show()

    def plot_weights_distribution(self, epoch):
        plt.figure(figsize=(10, 5))
        
        for i, w in enumerate(self.weights):
            plt.subplot(1, len(self.weights), i + 1)
            sns.histplot(w.flatten(), kde=True, bins=30, color="royalblue")
            plt.title(f"Weight Distribution Layer {i+1}")
            plt.xlabel("Weight values")
            plt.ylabel("Frequency")

        plt.suptitle(f"Weight Distributions at Epoch {epoch}")
        plt.tight_layout()
        plt.show()

    def predict(self, X):
        return self.forward(X)
    
    def MSE(self, X, Y):
        return np.mean((self.predict(X) - Y) ** 2)