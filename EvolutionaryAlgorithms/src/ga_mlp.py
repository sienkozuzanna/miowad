import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import f1_score
from tqdm import tqdm



class MLP:
    def __init__(self, layers, activation, last_activation):
        '''
        :param layers: A list specifying the number of neurons in each layer.
        :param activation: Activation function in hidden layers.
        :param last_activation: Activation function in output layer - for regression should be linear, 
                                for classification softmax
        '''
        self.layers = layers
        self.activation_function = activation
        self.last_layer_activation_function = last_activation
        self.weights = []
        self.bias = []

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
    
    def set_weights_from_vector(self, vector):
        self.weights, self.bias = [], []
        idx = 0
        for i in range(len(self.layers) - 1):
            in_size = self.layers[i]
            out_size = self.layers[i + 1]
            w = vector[idx:idx + in_size * out_size].reshape((in_size, out_size))
            idx += in_size * out_size
            b = vector[idx:idx + out_size]
            idx += out_size
            self.weights.append(w)
            self.bias.append(b)

    def weights_count(self):
        return sum((self.layers[i] + 1) * self.layers[i+1] for i in range(len(self.layers) - 1))



class GeneticAlgorithmMLP:
    def __init__(self, layers, X_train, Y_train, X_test=None, Y_test=None, 
                 population_size=50, generations=100, mutation_rate=0.1, crossover_rate=0.8, 
                 activation_function='relu', task='regression', scaler_X=None, scaler_Y=None):
        '''
        :param layers: list of int, specifying the number of neurons in each layer.
        :param X_train: np.ndarray, training input data.
        :param Y_train: np.ndarray, training target data.
        :param X_test: np.ndarray or None, test input data (optional).
        :param Y_test: np.ndarray or None, test target data (optional).
        :param population_size: int, number of individuals in the population (default: 50).
        :param generations: int, number of generations to run the algorithm (default: 100).
        :param mutation_rate: float, probability of mutation per gene (default: 0.1).
        :param crossover_rate: float, probability of crossover between parents (default: 0.8).
        :param activation_function: str, activation function name to use (default: 'relu').
        :param task: str, task type, either 'regression' or 'classification' (default: 'regression').
        '''
        
        self.layers = layers
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.task = task
        self.weights_count = self.compute_weights_count()
        self.scaler_X = scaler_X
        self.scaler_Y = scaler_Y

        self.activation_functions = {
            "sigmoid": lambda x: 1 / (1 + np.exp(-x)),
            "tanh": np.tanh,
            "relu": lambda x: np.maximum(0, x),
            "leaky_relu": lambda x: np.where(x > 0, x, 0.01 * x),
            "linear": lambda x: x,
            "softmax": lambda x: np.exp(x - np.max(x, axis=1, keepdims=True)) / np.sum(np.exp(x - np.max(x, axis=1, keepdims=True)), axis=1, keepdims=True)
        }

        if activation_function in self.activation_functions:
            self.activation_function = self.activation_functions[activation_function]
        else:
            raise ValueError(f"Unknown activation function: {activation_function}")

        if self.task == 'regression':
            self.last_activation_function = self.activation_functions['linear']
            self.loss_function = lambda y_pred, y_true: np.mean((y_pred - y_true) ** 2)  # MSE
            self.loss_name = 'MSE'
        elif self.task == 'classification':
            self.last_activation_function = self.activation_functions['softmax']
            self.loss_function = self.cross_entropy_loss
            self.loss_name = 'Cross Entropy Loss'
        else:
            raise ValueError("Task must be 'regression' or 'classification'")
        
        self.population = self.initialize_population()
        

    def cross_entropy_loss(self, y_pred, y_true):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
    
    def calculate_f1_score(self, individual, data='train'):
        if data == 'train':
            X, Y = self.X_train, self.Y_train
        elif data == 'test':
            if self.X_test is None or self.Y_test is None:
                return None
            X, Y = self.X_test, self.Y_test
        else:
            raise ValueError("data must be 'train' or 'test'")

        mlp = MLP(self.layers, self.activation_function, self.last_activation_function)
        mlp.set_weights_from_vector(individual)
        y_pred = mlp.forward(X)
        y_pred_labels = np.argmax(y_pred, axis=1)
        y_true_labels = np.argmax(Y, axis=1)
        f1 = f1_score(y_true_labels, y_pred_labels, average='macro')
        return f1
    
    def compute_weights_count(self):
        return sum((self.layers[i] + 1) * self.layers[i + 1] for i in range(len(self.layers) - 1))
    
    def initialize_population(self):
        population = []
        for _ in range(self.population_size):
            individual = np.random.uniform(-1, 1, self.weights_count)
            population.append(individual)
        return population
    
    def fitness_function(self, individual):
        mlp = MLP(self.layers, self.activation_function, self.last_activation_function)
        mlp.set_weights_from_vector(individual)
        y_pred = mlp.forward(self.X_train)
        if self.scaler_Y is not None:
            y_pred_orig = self.scaler_Y.inverse_transform(y_pred)
            y_true_orig = self.scaler_Y.inverse_transform(self.Y_train)
            return self.loss_function(y_pred_orig, y_true_orig)
        else:
            return self.loss_function(y_pred, self.Y_train)
    
    def evaluate_population(self):
        fitness_function_values=np.array([self.fitness_function(individual) for individual in self.population])
        return fitness_function_values
    
    def select_parents(self, fitness_values, tournament_size=3):
        parents=[]
        for _ in range(self.population_size):
            parents.append(self.tournament_selection(fitness_values, tournament_size))
        return parents
    
    def tournament_selection(self, fitness_values, tournament_size=3):
        selected_indices=np.random.choice(self.population_size, tournament_size, replace=False)
        selected_fitness_values=fitness_values[selected_indices]
        winner_index=np.argmin(selected_fitness_values)
        return self.population[selected_indices[winner_index]]
    
    def crossover(self, parent1, parent2):
        if np.random.rand() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        point = np.random.randint(1, len(parent1))
        child1 = np.concatenate([parent1[:point], parent2[point:]])
        child2 = np.concatenate([parent2[:point], parent1[point:]])
        return child1, child2
    
    def mutate(self, individual):
        if np.random.random() > self.mutation_rate:
            return individual[:]
        for i in range(len(individual)):
            if np.random.rand() < self.mutation_rate:
                individual[i] += np.random.normal(0, 1)
        return individual
    
    def evaluate_on_test(self, individual):
        if self.X_test is None or self.Y_test is None:
            return None 
        mlp = MLP(self.layers, self.activation_function, self.last_activation_function)
        mlp.set_weights_from_vector(individual)
        y_pred = mlp.forward(self.X_test)
        if self.scaler_Y is not None:
            y_pred_orig = self.scaler_Y.inverse_transform(y_pred)
            y_true_orig = self.scaler_Y.inverse_transform(self.Y_test)
            return self.loss_function(y_pred_orig, y_true_orig)
        else:
            return self.loss_function(y_pred, self.Y_test)
    
    def run(self, eval_every=100, early_stopping=False, patience=100, elitism_ratio=0.1):
        best_individual = None
        best_fitness = np.inf
        no_improvement_counter = 0
        self.best_fitness_history = []
        self.f1_history = []

        elitism = max(1, int(len(self.population) * elitism_ratio))

        for generation in range(self.generations):
            fitness_values = self.evaluate_population()

            elite_indices = np.argsort(fitness_values)[:elitism]
            elite_individuals = np.array(self.population)[elite_indices]

            selected_parents = self.select_parents(fitness_values)

            new_population = []
            for i in range(0, len(selected_parents), 2):
                if i + 1 >= len(selected_parents):  # if odd number of parents
                    break
                parent1, parent2 = selected_parents[i], selected_parents[i + 1]
                # crossover
                child1, child2 = self.crossover(parent1, parent2)
                # mutation
                child1, child2 = self.mutate(child1), self.mutate(child2)
                new_population.append(child1)
                new_population.append(child2)

            self.population = np.array(new_population[:len(self.population) - elitism] + list(elite_individuals))

            current_fitness = self.evaluate_population()
            current_best_index = np.argmin(current_fitness)
            current_best_fitness = current_fitness[current_best_index]

            if current_best_fitness < best_fitness:
                best_fitness = current_best_fitness
                best_individual = self.population[current_best_index].copy()
                no_improvement_counter = 0
            else:
                no_improvement_counter += 1

            self.best_fitness_history.append(best_fitness)
         
            if self.task == 'classification':
                if best_individual is not None:
                    f1_train = self.calculate_f1_score(best_individual, data='train')
                else:
                    f1_train = np.nan 
                self.f1_history.append(f1_train)


            if eval_every > 0 and (generation + 1) % eval_every == 0:
                print(f"Generation {generation + 1}/{self.generations}, Best Fitness: {best_fitness:.4f}")

            if early_stopping and no_improvement_counter >= patience:
                print(f"Early stopping at generation {generation + 1} — no improvement in {patience} generations.")
                break

            test_fitness = self.evaluate_on_test(best_individual) if best_individual is not None else None

        if test_fitness is not None:
            print(f"Best Fitness ({self.loss_name}) on Train set : {best_fitness:.4f}")
            print(f"Fitness ({self.loss_name}) on Test set: {test_fitness:.4f}")
            if self.task == 'classification':
                f1_train =  self.calculate_f1_score(best_individual, data='train')
                f1_test = self.calculate_f1_score(best_individual, data='test')
                if f1_test is not None:
                    print(f"F1-score on Train set: {f1_train:.4f}")
                    print(f"F1-score on Test set: {f1_test:.4f}")

        if self.task == 'regression':
            return best_individual, best_fitness, self.best_fitness_history
        else:
            return best_individual, best_fitness, self.best_fitness_history, self.f1_history
        
    def predict(self, individual, X):
        mlp = MLP(self.layers, self.activation_function, self.last_activation_function)
        mlp.set_weights_from_vector(individual)
        return mlp.forward(X)

    
    def plot_fitness_history(self, fitness_history, f1_history=None):
        fitness_history = np.array(fitness_history)
        generations = np.arange(len(fitness_history))

        if self.task == 'regression':
            plt.figure(figsize=(6, 6))
            plt.plot(generations, fitness_history, label=self.loss_name, color='blue')
            plt.title('Fitness History')
            plt.xlabel('Generations')
            plt.ylabel(self.loss_name)
            plt.grid(True)
            plt.legend()
            plt.show()

        elif self.task == 'classification':
            f1_history = np.array(f1_history) if f1_history is not None else np.full_like(fitness_history, np.nan)
            
            plt.figure(figsize=(12, 5))

            plt.subplot(1, 2, 1)
            plt.plot(generations, fitness_history, label=self.loss_name, color='blue')
            plt.title('Cross Entropy Loss History')
            plt.xlabel('Generations')
            plt.ylabel(self.loss_name)
            plt.grid(True)
            plt.legend()

            plt.subplot(1, 2, 2)
            plt.plot(generations, f1_history, label='F1 Score', color='green')
            plt.title('F1 Score History')
            plt.xlabel('Generations')
            plt.ylabel('F1 Score')
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.show()


    def plot_test_results(self, individual):
        if self.X_test is None or self.Y_test is None:
            print("No test data provided.")
            return
        
        mlp = MLP(self.layers, self.activation_function, self.last_activation_function)
        mlp.set_weights_from_vector(individual)
        y_pred = mlp.forward(self.X_test)

        if self.scaler_X is not None:
            X_plot = self.scaler_X.inverse_transform(self.X_test)
        else:
            X_plot = self.X_test

        if self.scaler_Y is not None:
            y_pred_plot = self.scaler_Y.inverse_transform(y_pred)
            y_true_plot = self.scaler_Y.inverse_transform(self.Y_test)
        else:
            y_pred_plot = y_pred
            y_true_plot = self.Y_test

        if self.task == 'regression':
            plt.figure(figsize=(6, 4))
            plt.scatter(X_plot, y_pred_plot, color='r', label="Predicted values")
            plt.scatter(X_plot, y_true_plot, color='b', label="True values")
            plt.xlabel('X_test')
            plt.ylabel('Y')
            plt.legend(loc='lower right')
            plt.title('Fitted vs Actual values for test data')
            plt.grid(True)
            plt.show()

        elif self.task == 'classification':
            y_true = np.argmax(self.Y_test, axis=1) if self.Y_test.ndim > 1 else self.Y_test
            y_pred_labels = np.argmax(y_pred, axis=1)

            cm = confusion_matrix(y_true, y_pred_labels)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot(cmap=plt.cm.Blues)
            plt.title('Confusion Matrix on Test Set')
            plt.show()

                