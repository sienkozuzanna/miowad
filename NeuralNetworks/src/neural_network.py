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

    def backpropagate(self, X, y):
        m = X.shape[0]
        delta = (self.forward(X) - y)/m 
        
        for i in reversed(range(len(self.weights))):
            delta_weights = np.dot(self.a[i].T, delta)
            delta_bias = np.sum(delta, axis=0)
            
            self.weights[i] -= self.learning_rate * delta_weights
            self.bias[i] -= self.learning_rate * delta_bias
            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * self.activation_function_derivative(self.a[i])


    def train(self, X_train, y_train, X_test, y_test, epochs, batch_size=32, verbose=True, show_training=True, weights_visualization_interval=1000):
        batch_size = batch_size if batch_size is not None else self.batch_size
        train_losses = []
        test_losses=[]
        start_time = time.time()
        
        for epoch in range(epochs):
            if batch_size is None: #full batch (all datasets records are used for computing gradient)
                self.forward(X_train)
                self.backpropagate(X_train, y_train)
            else:
                permutation = np.random.permutation(X_train.shape[0]) #mini-batch
                for i in range(0, X_train.shape[0], batch_size):
                    indices = permutation[i:i+batch_size]
                    X_batch = X_train[indices]
                    y_batch = y_train[indices]
                    self.forward(X_batch)
                    self.backpropagate(X_batch, y_batch)
            

            train_loss = self.MSE(X_train, y_train)
            test_loss = self.MSE(X_test, y_test)
            train_losses.append(train_loss)
            test_losses.append(test_loss)

            if verbose:
                if epoch % 100 == 0:
                    print(f"Epoch {epoch}, Training Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}")
        
            if show_training:
                if (epoch+1) % weights_visualization_interval == 0:
                    self.plot_weights_distribution(epoch+1)
            
        
        end_time = time.time()
        print(f"Total training time: {end_time - start_time:.2f} seconds")

        plt.figure(figsize=(6, 4))
        plt.plot(train_losses, label="Training Loss", color="blue", linewidth=2)
        plt.plot(test_losses, label="Test Loss", color="red", linestyle="--", linewidth=2)
        plt.title("Training and Test Losses")
        plt.xlabel("Epoch")
        plt.ylabel("MSE")
        plt.legend()
        plt.grid(True)
        plt.show()
    

    def predict(self, X):
        return self.forward(X)
    
    def MSE(self, X, Y):
        return np.mean((self.predict(X) - Y) ** 2)
    
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