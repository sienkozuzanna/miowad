import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
import pandas as pd
import plotly.graph_objects as go
from sklearn.manifold import TSNE
import seaborn as sns
from sklearn.metrics import confusion_matrix
import logging
from matplotlib.patches import RegularPolygon
from matplotlib.collections import PatchCollection

class KohonenNetwork:
    def __init__(self, M, N, input_dim, neighbourhood_function='gaussian', topology='rectangular'):
        self.M=M
        self.N=N
        self.input_dim = input_dim
        self.topology=topology #rectangular or hexagonal
        #each neuron in 2D grid has weight vector equal to number of inputs in data
        self.weights = np.random.rand(M, N, input_dim)

        if neighbourhood_function == 'gaussian':
            self.neighbourhood_function = self.gaussian_neighbourhood
        elif neighbourhood_function=='mexican_hat':
            self.neighbourhood_function = self.mexican_hat_neighbourhood
        else:
            raise ValueError("Invalid neighbourhood function. Choose 'gaussian' or 'mexican_hat'.")
        
    def initialize_weights_grid(self, data):
        min_vals = np.min(data, axis=0)
        max_vals = np.max(data, axis=0)
        self.weights = np.random.uniform(min_vals, max_vals, 
                                    size=(self.M, self.N, self.input_dim))

    def gaussian_neighbourhood(self, bmu, neuron, sigma, s=1.0):
        distance = self.calculate_distance(bmu, neuron)
        return np.exp(- (s * distance) ** 2 / (2 * sigma ** 2))

    #negative_second_derivative_gaussian
    def mexican_hat_neighbourhood(self, bmu, neuron, sigma, s=1.0):
        distance = self.calculate_distance(bmu, neuron)
        dist_scaled = s * distance
        return (1 - (dist_scaled ** 2 / sigma ** 2)) * np.exp(-dist_scaled ** 2 / (2 * sigma ** 2))
    
    def calculate_distance(self, bmu, neuron):
        if self.topology=='rectangular':
           return np.linalg.norm(np.array(bmu) - np.array(neuron))
        elif self.topology=='hexagonal':
            i1, j1 = bmu
            i2, j2 = neuron
            x1 = j1 - 0.5 * (i1 % 2)
            y1 = i1 * (np.sqrt(3)/2)
            x2 = j2 - 0.5 * (i2 % 2)
            y2 = i2 * (np.sqrt(3)/2)
            return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        else:
            raise ValueError('Invalid topology, try rectangular or hexagonal.')

    
    #function to decay learning rate while training
    def learning_rate_decay(self,lambda_, t):
        return np.exp(-t/lambda_)
    
    #neighbourhood function decay
    def sigma_decay(self, sigma_0, t, lambda_):
        return sigma_0 * np.exp(-t / lambda_)
    
    def find_bmu(self, x):
        distances = np.linalg.norm(self.weights - x.reshape(1,1,-1), axis=2)
        return np.unravel_index(np.argmin(distances), distances.shape)

    def fit(self, data, number_of_iterations, lambda_, sigma_t=1.0, s=1.0, 
            plot_eval_metrics=True, eval_every=10, initialize_weights=True, verbose=False):
        '''
        :param: data: input data
        :param: number_of_iterations: number of iterations for training
        :param: lambda_: decay constant for learning rate
        :param: sigma_t: initial neighbourhood size
        :param: s: scaling factor for neighbourhood function
        '''
        if verbose:
            logging.basicConfig(level=logging.INFO, format='%(message)s')
        else:
            logging.basicConfig(level=logging.WARNING)

        sigma_0=sigma_t
        data = np.array(data)
        if initialize_weights:
            self.initialize_weights_grid(data)
        quantization_errors, silhouette_scores, db_indexes, iterations  = [],[],[],[]

        for t in range(number_of_iterations):
            data_shuffle = data.copy()
            random.shuffle(data_shuffle)
            alpha_t=self.learning_rate_decay(lambda_, t)
            sigma = sigma_0 * np.exp(-t / lambda_)
            for x in data_shuffle:
                i_bmu=self.find_bmu(x)
                for i in range(self.M):
                    for j in range(self.N):
                        neighbourhood = self.neighbourhood_function(i_bmu, (i, j), sigma, s)
                        self.weights[i][j]+=neighbourhood*alpha_t*(x-self.weights[i][j])

            if plot_eval_metrics is True and t % eval_every == 0:
                quantization_errors.append(self.quantization_error(data))
                silhouette_scores.append(self.silhouette_score(data))
                db_indexes.append(self.davies_bouldin_score(data))
                iterations.append(t)

                if verbose:
                    logging.info(f'Iteration {t}:')
                    logging.info(f'  Learning Rate (alpha): {alpha_t:.4f}')
                    logging.info(f'  Neighborhood Size (sigma): {sigma:.4f}')
                    logging.info(f'  Quantization Error: {f"{quantization_errors[-1]:.4f}" if quantization_errors[-1] is not None else "nan"}')
                    logging.info(f'  Silhouette Score: {f"{silhouette_scores[-1]:.4f}" if silhouette_scores[-1] is not None else "nan"}')
                    logging.info(f'  Davies-Bouldin Index: {f"{db_indexes[-1]:.4f}" if db_indexes[-1] is not None else "nan"}')

        if plot_eval_metrics:
            self.plot_eval_metrics(iterations, silhouette_scores, db_indexes, quantization_errors)
        else:
            quantization_errors.append(self.quantization_error(data))
            silhouette_scores.append(self.silhouette_score(data))
            db_indexes.append(self.davies_bouldin_score(data))
            iterations.append(t)

    def predict(self, x):
        return self.find_bmu(x)
    
    def return_weights(self):
        return self.weights
    
    #getting only active neurons
    def get_cluster_labels(self, data):
        bmu_indices = [self.find_bmu(x) for x in data]
        unique_neurons = set(bmu_indices)
        return [list(unique_neurons).index((i,j)) for (i,j) in bmu_indices]
    
    def get_number_of_clusters(self, data):
        bmu_indices = [self.find_bmu(x) for x in data]
        return len(set(bmu_indices))
    
    def silhouette_score(self, data):
        labels = self.get_cluster_labels(data)
        if len(set(labels)) < 2:
            print('Only 1 cluster detected')
            return None
        return silhouette_score(data, labels)
    
    def davies_bouldin_score(self, data):
        labels = self.get_cluster_labels(data)
        if len(set(labels)) < 2:
            return None 
        return davies_bouldin_score(data, labels)
    
    #distance of data points to their bmu (best matching unit)
    def quantization_error(self, data):
        errors = [np.linalg.norm(point - self.weights[self.find_bmu(point)]) for point in data]
        return np.mean(errors)
    
    def calculate_clustering_metrics(self, data, true_labels=None):
        metrics = {
            "Silhouette Score": [self.silhouette_score(data)],
            "Davies-Bouldin Index": [self.davies_bouldin_score(data)],
            "Quantization Error": [self.quantization_error(data)],
            "Number of Clusters": [self.get_number_of_clusters(data)]
        }

        df_metrics = pd.DataFrame(metrics).round(2)
        return df_metrics
    
    def class_distribution_in_neurons(self, data, true_labels, return_dominant=False):
        true_labels = np.asarray(true_labels).flatten()
        bmu_indices = np.array([self.find_bmu(x) for x in data])
        class_dist = {(i,j): {} for i in range(self.M) for j in range(self.N)}
       
        for (i,j), label in zip(bmu_indices, true_labels):
            if label in class_dist[(i,j)]:
                class_dist[(i,j)][label] += 1
            else:
                class_dist[(i,j)][label] = 1

        if return_dominant:
            dominant_labels = {}
            for neuron in class_dist:
                if class_dist[neuron]:
                    dominant = max(class_dist[neuron].items(), key=lambda x: x[1])[0]
                    dominant_labels[neuron] = dominant
                else:
                    dominant_labels[neuron] = None
            return dominant_labels
        else:
            return class_dist
                
    
    def neuron_purity(self, data, true_labels):
        true_labels = np.asarray(true_labels).flatten()
        bmu_indices = np.array([self.find_bmu(x) for x in data])
        purity = {}
        neuron_counts = {}

        for i in range(self.M):
            for j in range(self.N):
                neuron_counts[(i, j)] = {}
                purity[(i, j)] = 0.0
        
        for (i, j), label in zip(bmu_indices, true_labels):
            if label in neuron_counts[(i, j)]:
                neuron_counts[(i, j)][label] += 1
            else:
                neuron_counts[(i, j)][label] = 1
        
        for i in range(self.M):
            for j in range(self.N):
                counts = list(neuron_counts[(i, j)].values())
                if counts:
                    purity[(i, j)] = max(counts) / sum(counts)
                else:
                    purity[(i, j)] = 0.0
                    
        return purity
    
    def get_neurons_defining_class(self, data, true_labels):
        true_labels = np.asarray(true_labels).flatten()
        bmu_indices = [self.find_bmu(x) for x in data]

        #mapping each class to neurons
        class_to_neurons = {}
        for (i,j), label in zip(bmu_indices, true_labels):
            if label not in class_to_neurons:
                class_to_neurons[label] = set()
            class_to_neurons[label].add((i,j))

        #neuron purity and dominant class
        neuron_stats = {}
        neuron_counts = {(i,j): {} for i in range(self.M) for j in range(self.N)}
        for (i,j), label in zip(bmu_indices, true_labels):
            neuron_counts[(i,j)][label] = neuron_counts[(i,j)].get(label, 0) + 1
        
        for (i,j), counts in neuron_counts.items():
            if counts:
                dominant_class = max(counts.items(), key=lambda x: x[1])[0]
                purity = counts[dominant_class] / sum(counts.values())
                neuron_stats[(i,j)] = (dominant_class, purity)
            else:
                neuron_stats[(i,j)] = (None, 0.0)
        
        return class_to_neurons, neuron_stats
    
    def compute_umat(self):
        umat = np.zeros((self.M, self.N))
        
        for i in range(self.M):
            for j in range(self.N):
                distances = []
                if i > 0:
                    distances.append(np.linalg.norm(self.weights[i][j] - self.weights[i-1][j]))
                if i < self.M - 1:
                    distances.append(np.linalg.norm(self.weights[i][j] - self.weights[i+1][j]))
                if j > 0:
                    distances.append(np.linalg.norm(self.weights[i][j] - self.weights[i][j-1]))
                if j < self.N - 1:
                    distances.append(np.linalg.norm(self.weights[i][j] - self.weights[i][j+1]))
                
                umat[i][j] = np.mean(distances) if distances else 0
        
        return umat
            
#--------------------------------VISUALIZATIONS-------------------------------

    def plot_eval_metrics(self, iterations, silhouette_scores, db_indexes, quantization_errors):
        fig, axs = plt.subplots(1, 3, figsize=(12, 4), sharex=False)
        axs[0].plot(iterations, quantization_errors, label='Quantization Error', color='blue')
        axs[0].set_ylabel("Quantization Error")
        axs[0].grid(True)
        axs[0].legend()

        axs[1].plot(iterations, silhouette_scores, label='Silhouette Score', color='blue')
        axs[1].set_ylabel("Silhouette Score")
        axs[1].grid(True)
        axs[1].legend()

        axs[2].plot(iterations, db_indexes, label='Davies-Bouldin Index', color='blue')
        axs[2].set_ylabel("Davies-Bouldin Index")
        axs[2].set_xlabel("Iteration")
        axs[2].grid(True)
        axs[2].legend()

        plt.tight_layout()
        plt.show()

    def plot_umatrix(self, data, true_labels):
        umat = self.compute_umat()
        fig = plt.figure(figsize=(12, 6))
        
        if hasattr(self, 'neighbourhood_function'):
            if self.neighbourhood_function == self.gaussian_neighbourhood:
                ax = fig.add_subplot(121)
                title = "Gaussian Neighbourhood"
            elif self.neighbourhood_function == self.mexican_hat_neighbourhood:
                ax = fig.add_subplot(122)
                title = "Mexican Hat Neighbourhood"
            else:
                ax = fig.add_subplot(111)
                title = "SOM U-Matrix (Hexagonal Grid)"
        else:
            ax = fig.add_subplot(111)
            title = "SOM U-Matrix (Hexagonal Grid)"

        dominant_labels = self.class_distribution_in_neurons(data, true_labels, return_dominant=True)

        if self.topology == 'hexagonal':
            patches, values = [], []
            radius = 0.5
            dx = 1.5 * radius
            dy = np.sqrt(3) * radius

            for i in range(self.M):
                for j in range(self.N):
                    x = j * dx + (0.5 * dx if i % 2 == 1 else 0) 
                    y = i * dy
                    hexagon = RegularPolygon((x, y), numVertices=6, radius=radius,
                                            orientation=np.radians(0), edgecolor='k')
                    patches.append(hexagon)
                    values.append(umat[i, j])

                    dominant_class = dominant_labels.get((i, j), None)
                    if dominant_class is not None:
                        ax.text(x, y, str(dominant_class), color='black',
                                ha='center', va='center', fontsize=10)

            pc = PatchCollection(patches, cmap='viridis', alpha=0.8)
            pc.set_array(np.array(values))
            ax.add_collection(pc)

            max_x = (self.N - 1) * dx
            if self.M > 1:
                max_x += 0.5 * dx 
            ax.set_aspect('equal')
            ax.set_xlim(-radius, max_x + radius)
            ax.set_ylim(-radius, (self.M - 0.5) * dy + radius)
            ax.set_xticks(np.arange(0, self.N * dx, dx))
            ax.set_xticklabels(np.arange(self.N))
            ax.set_yticks(np.arange(0, self.M * dy, dy))
            ax.set_yticklabels(np.arange(self.M))
            ax.set_title(title)
            ax.set_xlabel("Neuron Column Index")
            ax.set_ylabel("Neuron Row Index")

        else:
            im = ax.imshow(umat, cmap='viridis', interpolation='nearest')
            for i in range(self.M):
                for j in range(self.N):
                    dominant_class = dominant_labels.get((i, j), None)
                    if dominant_class is not None:
                        ax.text(j, i, str(dominant_class), color='black',
                                ha='center', va='center', fontsize=10)
            ax.set_title("SOM U-Matrix (Rectangular Topology)")

        cbar = fig.colorbar(pc if self.topology == 'hexagonal' else im, ax=ax)
        cbar.set_label("Mean distance between neurons", rotation=270, labelpad=15)
        plt.tight_layout()
        plt.show()

    def visualize_neurons_2d(self):
        if self.input_dim != 2:
            raise ValueError("2D visualization requires input_dim=2")
        
        plt.figure(figsize=(6, 6))
        if self.topology == 'rectangular':
            for i in range(self.M):
                for j in range(self.N):
                    plt.scatter(self.weights[i][j][0], self.weights[i][j][1], 
                             c='blue', edgecolors='gray', marker='o', s=100, 
                             label='Neurons' if (i == 0 and j == 0) else "")
                    if j + 1 < self.N:
                        plt.plot([self.weights[i][j][0], self.weights[i][j+1][0]],
                                 [self.weights[i][j][1], self.weights[i][j+1][1]], 
                                 'gray', linestyle='-', linewidth=1, alpha=0.5)
                    if i + 1 < self.M:
                        plt.plot([self.weights[i][j][0], self.weights[i+1][j][0]],
                                 [self.weights[i][j][1], self.weights[i+1][j][1]], 
                                 'gray', linestyle='-', linewidth=1, alpha=0.5)
        
        elif self.topology == 'hexagonal':
            offset = 0.5 
            for i in range(self.M):
                for j in range(self.N):
                    x = j - (i % 2) * offset
                    y = i * 0.866  # sqrt(3)/2
                    
                    plt.scatter(self.weights[i][j][0], self.weights[i][j][1], 
                             c='blue', edgecolors='gray', marker='o', s=100, 
                             label='Neurons' if (i == 0 and j == 0) else "")
                    if j + 1 < self.N:
                        plt.plot([self.weights[i][j][0], self.weights[i][j+1][0]],
                                 [self.weights[i][j][1], self.weights[i][j+1][1]], 
                                 'gray', linestyle='-', linewidth=1, alpha=0.5)
                    if i + 1 < self.M and (j + (i % 2)) < self.N:
                        plt.plot([self.weights[i][j][0], self.weights[i+1][j + (i % 2)][0]],
                                 [self.weights[i][j][1], self.weights[i+1][j + (i % 2)][1]], 
                                 'gray', linestyle='-', linewidth=1, alpha=0.5)
                    
                    if i + 1 < self.M and (j - ((i + 1) % 2)) >= 0:
                        plt.plot([self.weights[i][j][0], self.weights[i+1][j - ((i + 1) % 2)][0]],
                                 [self.weights[i][j][1], self.weights[i+1][j - ((i + 1) % 2)][1]], 
                                 'gray', linestyle='-', linewidth=1, alpha=0.5)

        plt.title(f'SOM Neurons (2D) - {self.topology.capitalize()} Topology')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()


    def visualize_clusters(self, data, true_labels=None):
        if self.input_dim != 2:
            raise ValueError("2D visualization is only available for 2D data (input_dim=2)")

        plt.figure(figsize=(6, 4))
        cluster_indices = np.array([self.find_bmu(point) for point in data])
        unique_clusters = cluster_indices[:, 0] * self.N + cluster_indices[:, 1]
       
        num_clusters = len(np.unique(unique_clusters))
        cmap = plt.get_cmap('tab10', num_clusters) if num_clusters <= 10 else plt.get_cmap('tab20', num_clusters)
        
        scatter = plt.scatter(data[:, 0], data[:, 1], c=unique_clusters, 
                            cmap=cmap, alpha=0.6, label='Data points')
        
        neurons_plotted = False
        for i in range(self.M):
            for j in range(self.N):
                if not neurons_plotted:
                    plt.scatter(self.weights[i][j][0], self.weights[i][j][1], 
                            c='black', marker='x', s=100, linewidth=2, label='Neurons')
                    neurons_plotted = True
                else:
                    plt.scatter(self.weights[i][j][0], self.weights[i][j][1], 
                            c='black', marker='x', s=100, linewidth=2)
                
                if j + 1 < self.N:
                    plt.plot([self.weights[i][j][0], self.weights[i][j+1][0]],
                            [self.weights[i][j][1], self.weights[i][j+1][1]], 
                            c='gray', alpha=0.3)
                if i + 1 < self.M:
                    plt.plot([self.weights[i][j][0], self.weights[i+1][j][0]],
                            [self.weights[i][j][1], self.weights[i+1][j][1]], 
                            c='gray', alpha=0.3)
        
        if true_labels is not None:
            true_scatter = plt.scatter(data[:, 0], data[:, 1], c=true_labels, 
                                    cmap='Pastel1', alpha=0.3, marker='.', 
                                    label='True clusters (if available)')
        
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        
        plt.title('SOM Cluster Visualization')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid(True)
        plt.colorbar(scatter, label='Cluster ID')
        plt.tight_layout()
        plt.show()

    def visualize_clusters_3d(self, data, true_labels=None):
        fig = go.Figure()

        clusters = [self.find_bmu(p)[0]*self.N + self.find_bmu(p)[1] for p in data]
        fig.add_trace(go.Scatter3d(x=data[:,0], y=data[:,1], z=data[:,2], mode='markers',
                                    marker=dict(size=4, color=clusters, opacity=0.7), name='Clusters'))

        weights_flat = self.weights.reshape(-1,3)
        fig.add_trace(go.Scatter3d(x=weights_flat[:,0], y=weights_flat[:,1], z=weights_flat[:,2], 
                                    mode='markers+lines', marker=dict(size=6, color='black', symbol='x'),
                                    line=dict(color='gray', width=1), name='Neurons'))

        if true_labels is not None:
            fig.add_trace(go.Scatter3d(x=data[:,0], y=data[:,1], z=data[:,2], mode='markers',
                                        marker=dict(size=3, color=true_labels, opacity=0.3), name='True Labels'))

        fig.update_layout(scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'), 
                            margin=dict(l=0,r=0,b=0,t=30))
        fig.show()


    def visualize_heatmap(self, data, true_labels=None, cmap='magma', grid=True):
        cluster_counts = np.zeros((self.M, self.N))
        cluster_labels = {}
        
        for idx, point in enumerate(data):
            i, j = self.find_bmu(point)
            cluster_counts[i][j] += 1
            if true_labels is not None:
                if (i,j) not in cluster_labels:
                    cluster_labels[(i,j)] = []
                cluster_labels[(i,j)].append(true_labels[idx])
        
        plt.figure(figsize=(6, 4))
        im = plt.imshow(cluster_counts, cmap=cmap, interpolation='nearest')
        if grid:
            plt.gca().set_xticks(np.arange(-0.5, self.N, 1), minor=True)
            plt.gca().set_yticks(np.arange(-0.5, self.M, 1), minor=True)
            plt.grid(which='minor', color='gray', linestyle='-', linewidth=1)
            plt.gca().tick_params(which='minor', size=0)
        
        plt.gca().set_xticks(np.arange(self.N))
        plt.gca().set_yticks(np.arange(self.M))
        cbar = plt.colorbar(im)
        cbar.set_label('Number of points in neuron', rotation=270, labelpad=20)
        
        plt.title('Neuron Activation Heatmap')
        plt.xlabel('Neuron Column Index')
        plt.ylabel('Neuron Row Index')
        
        if true_labels is not None:
            for (i,j), labels in cluster_labels.items():
                unique, counts = np.unique(labels, return_counts=True)
                dominant_label = unique[np.argmax(counts)]
                purity = np.max(counts)/np.sum(counts)
                
                cell_color = im.cmap(im.norm(cluster_counts[i,j]))
                text_color = 'white' if np.mean(cell_color[:3]) < 0.6 else 'black'
                
                plt.text(j, i, f"{dominant_label}\n{int(purity*100)}%", 
                        ha='center', va='center', 
                        color=text_color, fontsize=8)
        
        plt.tight_layout()
        plt.show()

    def visualize_tsne(self, data, true_labels=None, perplexity=30, n_iter=1000, random_state=42):
        tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=random_state)
        data_2d = tsne.fit_transform(data)

        plt.figure(figsize=(6, 4))
        if true_labels is not None:
            scatter = plt.scatter(data_2d[:, 0], data_2d[:, 1], c=true_labels, cmap='tab10', alpha=0.7)
            plt.legend(*scatter.legend_elements(), title="Classes")
        else:
            plt.scatter(data_2d[:, 0], data_2d[:, 1], alpha=0.7)
        
        plt.title('t-SNE Visualization of Data')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def visualize_tsne_neurons(self):
        weights_flat = self.weights.reshape(-1, self.input_dim)
        tsne = TSNE(n_components=2, perplexity=2, n_iter=1000)
        reduced_weights = tsne.fit_transform(weights_flat)

        plt.figure(figsize=(6, 4))
        plt.scatter(reduced_weights[:, 0], reduced_weights[:, 1], c='black', marker='x')
        plt.title("t-SNE Visualization of SOM Neurons")
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_confusion_matrix(self, data, true_labels, dominant_labels=None, figsize=(10, 8), cmap='Blues'):
        if dominant_labels is None:
            dominant_labels = self.class_distribution_in_neurons(data, true_labels, return_dominant=True)
        
        pred_labels = []
        for x in data:
            neuron = self.find_bmu(x)
            pred_labels.append(dominant_labels[neuron])
        
        cm = confusion_matrix(true_labels, pred_labels)
        
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap=cmap)
        plt.title('Confusion Matrix: True Classes vs Neuron-Dominant Classes')
        plt.xlabel('Predicted (Neuron Dominant Class)')
        plt.ylabel('True Class')
        plt.tight_layout()
        plt.show()
    
    

    


    
