import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random

class CuttingStockAlgorithm:
    def __init__(self, r, rectangles, population_size=100, 
                mutation_rate=0.1, crossover_rate=0.7, selection_method='tournament'):
        
        '''
        Initializes the Cutting Stock Algorithm parameters.
        Parameters:
        - r: int, radius of circle
        - rectangles: list of tuples, each tuple contains (height, width, value)
        - population_size: int, size of the population for the genetic algorithm

        - mutation_rate: float, probability of mutation in the genetic algorithm
        - crossover_rate: float, probability of crossover in the genetic algorithm
        - selection_method: str, method for selecting parents ('tournament' or 'roulette')
        '''
        
        self.r=r
        self.rectangles = np.array(rectangles) #shape (n, 3)
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        if selection_method not in ['tournament', 'roulette', 'rank']:
            raise ValueError("Selection method must in []'tournament''roulette', 'rank']")
        else:
            self.selection_method = selection_method
        self.population = self.initialize_population()

    def initialize_population(self):
        population = []
        for _ in range(self.population_size):
            individual = []
            for _ in range(200):
                rect_idx = np.random.randint(len(self.rectangles))
                h, w, _ = self.rectangles[rect_idx]
                rotated = np.random.choice([True, False])
                rh, rw = (w, h) if rotated else (h, w)
                layer = np.random.choice(["outer", "middle", "core"], p=[0.6, 0.3, 0.1])
                
                if layer == "outer":
                    radius = np.random.uniform(0.9 * self.r, self.r)
                    angle = np.random.uniform(0, 2 * np.pi)
                    x = radius * np.cos(angle)
                    y = radius * np.sin(angle)
                    x -= rw/2 * np.cos(angle)
                    y -= rh/2 * np.sin(angle)
                elif layer == "middle":
                    radius = np.random.uniform(0.6 * self.r, 0.9 * self.r)
                    angle = np.random.uniform(0, 2 * np.pi)
                    x = radius * np.cos(angle)
                    y = radius * np.sin(angle)
                else:
                    x = np.random.uniform(-0.6 * self.r, 0.6 * self.r)
                    y = np.random.uniform(-0.6 * self.r, 0.6 * self.r)
                
                corners = [(x - rw/2, y - rh/2), (x + rw/2, y - rh/2),
                        (x - rw/2, y + rh/2), (x + rw/2, y + rh/2)]
                if all(np.sqrt(cx**2 + cy**2) <= self.r for cx, cy in corners):
                    if not self.check_for_collision(x, y, rw, rh, individual):
                        individual.append((rect_idx, x, y, rotated))
            
            population.append(individual)
        return population
        
    def check_for_collision(self, x, y, w, h, individual):
        for rect_idx, rx, ry, r_rotated in individual:
            rh, rw, _ = self.rectangles[rect_idx]
            rh, rw = (rw, rh) if r_rotated else (rh, rw)
            dx, dy = abs(x - rx), abs(y - ry)
            if dx >= (w + rw) / 2 or dy >= (h + rh) / 2:
                continue
            return True
        return False
    
    def fitness(self, individual):
        if not individual:
            return 0
        total_value = 0
        for rect_index, _, _, _ in individual:
            total_value += self.rectangles[rect_index][2]
        return total_value
    
    def select_parents_tournament(self, tournament_size=3):
        competitors = random.sample(self.population, tournament_size)
        return max(competitors, key=self.fitness)
    
    def select_parents_roulette(self):
        fitness_values = np.array([self.fitness(ind) for ind in self.population])
        total_fitness = fitness_values.sum()
        if total_fitness == 0:
            return np.random.choice(self.population)
        probs = fitness_values / total_fitness
        selected_idx = np.random.choice(len(self.population), p=probs)
        return self.population[selected_idx]
    
    def select_parents_ranking(self):
        fitness_values = np.array([self.fitness(ind) for ind in self.population])
        sorted_indices = np.argsort(fitness_values)[::-1]
        ranks = np.empty_like(sorted_indices)
        ranks[sorted_indices] = np.arange(1, len(self.population) + 1)
        n = len(self.population)
        selection_probs = (n - ranks + 1) / (n * (n + 1) / 2)  # sum to 1
        selected_idx = np.random.choice(len(self.population), p=selection_probs)
        
        return self.population[selected_idx]

    def crossover(self, parent1, parent2):
        if np.random.random() > self.crossover_rate or not parent1 or not parent2:
            return parent1[:], parent2[:]
        
        min_length = min(len(parent1), len(parent2))
        if min_length == 0:
            return parent1[:], parent2[:]
        
        crossover_point = np.random.randint(1, min_length)
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        child1, child2 = self.validate_individual(child1), self.validate_individual(child2)
        return child1, child2  
    
    def validate_individual(self, individual):
        valid_individual = []
        for rect_idx, x, y, rotated in individual:
            h, w, _ = self.rectangles[rect_idx]
            rh, rw = (w, h) if rotated else (h, w)
            corners = [(x - rw/2, y - rh/2), (x + rw/2, y - rh/2),
                    (x - rw/2, y + rh/2), (x + rw/2, y + rh/2)]
            if all(np.sqrt(cx**2 + cy**2) <= self.r for cx, cy in corners):
                if not self.check_for_collision(x, y, rw, rh, valid_individual):
                    valid_individual.append((rect_idx, x, y, rotated))
        return valid_individual

    def mutate(self, individual, mutation_type='random'):
        if np.random.random() > self.mutation_rate:
            return individual[:]
        if mutation_type not in ['random', 'add', 'remove', 'reposition']:
            raise ValueError("Mutation type must be 'random', 'add', 'remove', 'reposition'")
        
        individual = individual.copy()
        if mutation_type == 'random':
            if not individual:
                mutation_type = 'add'
            else:
                mutation_type = np.random.choice(['add', 'remove', 'reposition'], p=[0.5, 0.2, 0.3])

        if mutation_type == 'add':
            rect_index = np.random.randint(len(self.rectangles))
            h, w, _ = self.rectangles[rect_index]
            if individual:
                placed, tries = False, 0
                while not placed and tries < 100:
                    ref_idx = np.random.randint(len(individual))
                    ref_rect = individual[ref_idx]
                    ref_x, ref_y = ref_rect[1], ref_rect[2]
                    ref_rotated = ref_rect[3]
                    ref_h, ref_w, _ = self.rectangles[ref_rect[0]]
                    ref_rh, ref_rw = (ref_w, ref_h) if ref_rotated else (ref_h, ref_w)
                
                    rotated = np.random.choice([True, False])
                    rh, rw = (w, h) if rotated else (h, w)
                    side = np.random.choice(['top', 'bottom', 'left', 'right'])
                    
                    if side == 'top':
                        x, y = ref_x, ref_y + ref_rh/2 + rh/2
                    elif side == 'bottom':
                        x, y = ref_x, ref_y - ref_rh/2 - rh/2
                    elif side == 'left':
                        x, y = ref_x - ref_rw/2 - rw/2, ref_y
                    else:
                        x, y = ref_x + ref_rw/2 + rw/2, ref_y
                    
                    corners = [(x - rw/2, y - rh/2), (x + rw/2, y - rh/2),
                            (x - rw/2, y + rh/2), (x + rw/2, y + rh/2)]
                    
                    if all(np.sqrt(cx**2 + cy**2) <= self.r for cx, cy in corners):
                        if not self.check_for_collision(x, y, rw, rh, individual):
                            individual.append((rect_index, x, y, rotated))
                            placed = True
                    tries += 1
                
                if not placed:
                    rotated = np.random.choice([True, False])
                    rh, rw = (w, h) if rotated else (h, w)
                    placed, tries = False, 0
                    while not placed and tries < 100:
                        layer = np.random.choice(["outer", "middle", "core"], p=[0.6, 0.3, 0.1])
                        if layer == "outer":
                            radius = np.random.uniform(0.9 * self.r, self.r)
                            angle = np.random.uniform(0, 2 * np.pi)
                            x, y = radius * np.cos(angle), radius * np.sin(angle)
                            x -= rw/2 * np.cos(angle)
                            y -= rh/2 * np.sin(angle)
                        elif layer == "middle":
                            radius = np.random.uniform(0.6 * self.r, 0.9 * self.r)
                            angle = np.random.uniform(0, 2 * np.pi)
                            x, y = radius * np.cos(angle), radius * np.sin(angle)
                        else:
                            x = np.random.uniform(-0.6 * self.r, 0.6 * self.r)
                            y = np.random.uniform(-0.6 * self.r, 0.6 * self.r)
                        
                        corners = [(x - rw/2, y - rh/2), (x + rw/2, y - rh/2),
                                (x - rw/2, y + rh/2), (x + rw/2, y + rh/2)]
                        if all(np.sqrt(cx**2 + cy**2) <= self.r for cx, cy in corners):
                            if not self.check_for_collision(x, y, rw, rh, individual):
                                individual.append((rect_index, x, y, rotated))
                                placed = True
                        tries += 1

        elif mutation_type == 'reposition' and individual:
            idx = np.random.randint(len(individual))
            rect_idx, old_x, old_y, rotated = individual[idx]
            h, w, _ = self.rectangles[rect_idx]
            rh, rw = (w, h) if rotated else (h, w)
            
            placed = False
            tries = 0
            while not placed and tries < 100:
                dx, dy = np.random.uniform(-0.5 * rw, 0.5 * rw), np.random.uniform(-0.5 * rh, 0.5 * rh)
                new_x, new_y = old_x + dx, old_y + dy
                corners = [(new_x - rw/2, new_y - rh/2), (new_x + rw/2, new_y - rh/2),
                        (new_x - rw/2, new_y + rh/2), (new_x + rw/2, new_y + rh/2)]
                
                if all(np.sqrt(cx**2 + cy**2) <= self.r for cx, cy in corners):
                    temp_individual = individual[:idx] + individual[idx+1:]
                    if not self.check_for_collision(new_x, new_y, rw, rh, temp_individual):
                        individual[idx] = (rect_idx, new_x, new_y, rotated)
                        placed = True
                tries += 1
                
        elif mutation_type == 'remove' and individual:
            individual.pop(np.random.randint(len(individual)))

        return individual

    def run(self, generations = 1000, eval_every=100, tournament_size=3, mutation_type='random', elitism=10):
        best_fitness, best_individual, fitness_history, best_individuals = -np.inf, None, [], []
        elitism = elitism
        
        for generation in range(generations):
            new_population = []
            
            for _ in range(self.population_size // 2):
                if self.selection_method == 'tournament':
                    parent1 = self.select_parents_tournament(tournament_size)
                    parent2 = self.select_parents_tournament(tournament_size)
                elif self.selection_method == 'rank':
                    parent1 = self.select_parents_ranking()
                    parent2 = self.select_parents_ranking()
                else:
                    parent1 = self.select_parents_roulette()
                    parent2 = self.select_parents_roulette()
    
                child1, child2 = self.crossover(parent1, parent2)
                child1, child2 = self.mutate(child1, mutation_type), self.mutate(child2, mutation_type)
                new_population.extend([child1, child2])
            
            sorted_population = sorted(self.population, key=self.fitness, reverse=True)
            elite = sorted_population[:elitism]
            self.population = elite + new_population[:-elitism]

            current_best = max(self.population, key=self.fitness)
            current_fitness = self.fitness(current_best)
            
            if current_fitness > best_fitness:
                best_fitness, best_individual = current_fitness, current_best
            fitness_history.append(best_fitness)
            best_individuals.append(best_individual)
            
            if generation % eval_every == 0:
                print(f"Generation {generation}, Best Fitness: {best_fitness}")
        
        return best_individual, best_fitness, fitness_history, best_individuals
        
#---------------------------------------------------------------------------------------------------------

    def plot_best_individual(self):
        individual = max(self.population, key=self.fitness, default=[])
        
        fig, ax = plt.subplots(figsize=(6, 6))
        circle = plt.Circle((0, 0), self.r, color='lightblue', fill=True, zorder=0)
        ax.add_patch(circle)

        total_area = np.pi * self.r ** 2
        used_area = 0

        for rect_idx, x, y, rotated in individual:
            h, w, _ = self.rectangles[rect_idx]
            rh, rw = (w, h) if rotated else (h, w)
            used_area += rh * rw
            lower_left_x, lower_left_y = x - rw / 2, y - rh / 2
            rect_patch = patches.Rectangle(
                (lower_left_x, lower_left_y), rw, rh,
                linewidth=1, edgecolor='black', facecolor='pink', alpha=0.7, zorder=1)
            ax.add_patch(rect_patch)

        utilization_percentage = (used_area / total_area) * 100

        ax.set_xlim(-self.r - 1, self.r + 1)
        ax.set_ylim(-self.r - 1, self.r + 1)
        ax.set_aspect('equal')
        ax.set_title(f"Best Rectangle Placement\nFitness: {self.fitness(individual)}\nSpace Utilization: {utilization_percentage:.2f}%")
        plt.grid(True)
        plt.show()

    def plot_fitness_history(self, fitness_history, target_value=None):
        fitness_history = np.array(fitness_history)
        generations = np.arange(len(fitness_history))

        plt.figure(figsize=(6, 6))
        plt.plot(generations, fitness_history, label='Best Fitness', color='blue')

        if target_value is not None:
            plt.axhline(target_value, color='red', linestyle='--', label='Target Value')

        plt.title('Fitness History')
        plt.xlabel('Generations')
        plt.ylabel('Fitness Value')
        plt.grid(True)
        plt.legend()
        plt.show()