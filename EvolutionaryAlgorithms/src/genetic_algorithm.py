import numpy as np

class GeneticAlgorithm:
    def __init__(self, dimension, population_size, fitness_function, mutation_rate, crossover_rate, generations, bounds=None):
        '''
        :param dimension: int, the number of dimensions of the problem
        :param population_size: int, the number of individuals in the population
        :param fitness_function: function, the fitness function to evaluate individuals
        :param mutation_rate: float, the probability of mutation
        :param crossover_rate: float, the probability of crossover
        :param generations: int, the number of generations to run the algorithm
        :param bounds: tuple, the lower and upper bounds for each dimension (optional)
        '''
        self.dimension=dimension
        self.population_size = population_size
        self.fitness_function=fitness_function
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.generations = generations
        self.bounds=bounds
        self.population=self.initialize_population()

    def initialize_population(self):
        if self.bounds is not None:
            lower_bound, upper_bound=self.bounds
            return np.random.uniform(lower_bound, upper_bound, (self.population_size, self.dimension))
        return np.random.rand(self.population_size, self.dimension)
    
    def evaluate_population(self):
        fitness_function_values=np.array([self.fitness_function(individual) for individual in self.population])
        return fitness_function_values
    
    #selecting parents using tournament selection by default
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
    
    def cross_over(self, parent1, parent2):
        #one-point crossover
        if np.random.rand()<self.crossover_rate:
            crossover_point=np.random.randint(1, self.dimension-1)
            child1=np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
            child2=np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
            return child1, child2
        return parent1, parent2
    
    #gaussian mutation
    def mutate(self, individual):
        if np.random.rand()<self.mutation_rate:
            mutation_vector=np.random.normal(0,1,self.dimension)
            individual += mutation_vector
            if self.bounds is not None:
                individual = np.clip(individual, self.bounds[0], self.bounds[1])
        return individual

    def run(self, eval_every=100, early_stopping=False, patience=100):
        best_individual = None
        best_fitness = np.inf
        no_improvement_counter = 0
        self.best_fitness_history = []

        for generation in range(self.generations):
            #stop condition
            #assume max_iterations=generations

            #evaluation of current population
            fitness_values=self.evaluate_population()
            selected_parents=self.select_parents(fitness_values)
            new_population=[]

            for i in range(0, len(selected_parents), 2):
                if i+1 >= len(selected_parents): #if odd number of parents
                    break
                parent1, parent2 = selected_parents[i], selected_parents[i+1]
                #crossover
                child1, child2=self.cross_over(parent1, parent2)
                #mutation
                child1, child2 =self.mutate(child1), self.mutate(child2)

                new_population.append(child1), new_population.append(child2)

            self.population=np.array(new_population)
            #evaluation of new population
            current_fitness = self.evaluate_population()
            current_best_index = np.argmin(current_fitness)
            current_best_fitness = current_fitness[current_best_index]
            self.best_fitness_history.append(current_best_fitness)

            #selecting best individual
            if current_best_fitness < best_fitness:
                best_fitness = current_best_fitness
                best_individual = self.population[current_best_index].copy()
                no_improvement_counter = 0
            else:
                no_improvement_counter += 1

            if eval_every > 0 and (generation + 1) % eval_every == 0:
                print(f"Generation {generation + 1}/{self.generations}, Best Fitness: {best_fitness}, Best Individual: {best_individual}")
            if early_stopping and no_improvement_counter >= patience:
                print(f"Early stopping at generation {generation + 1} — no improvement in {patience} generations.")
                break

        return best_individual, best_fitness, self.best_fitness_history

