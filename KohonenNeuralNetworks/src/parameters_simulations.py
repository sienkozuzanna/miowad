import kohonen_network
import importlib
importlib.reload(kohonen_network)
from kohonen_network import KohonenNetwork
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#different N, M sizes
def try_different_grid_sizes(data, M_values, N_values, input_dim, neighbourhood_function, 
                              number_of_iterations, lambda_, sigma_t, s):
    results = []

    for M in M_values:
        for N in N_values:
            som = KohonenNetwork(M=M, N=N, input_dim=input_dim, neighbourhood_function=neighbourhood_function)
            print(f'Fit with grid: {M}x{N}, number of neurons in grid: {M*N}')
            som.fit(data, number_of_iterations=number_of_iterations, lambda_=lambda_, 
                    sigma_t=sigma_t, s=s, plot_eval_metrics=False)
            
            quantization_error = som.quantization_error(data)
            silhouette = som.silhouette_score(data)
            db_index = som.davies_bouldin_score(data)
            num_clusters = som.get_number_of_clusters(data)
            results.append({
                'som': som,
                'M': M,
                'N': N,
                'number_of_neurons': M*N,
                'quantization_error': quantization_error,
                'silhouette_score': silhouette,
                'davies_bouldin_index': db_index,
                'num_clusters': num_clusters
            })

    return results


#different neighbourhood functions
def try_neighbourhood_functions(data, M, N, input_dim, number_of_iterations, lambda_, sigma_t, s):
    results = []
    neighbourhood_functions=['gaussian', 'mexican_hat']
    for function in neighbourhood_functions:
        som=KohonenNetwork(M=M, N=N, input_dim=input_dim, neighbourhood_function=function)
        som.fit(data, number_of_iterations=number_of_iterations, lambda_=lambda_, 
                sigma_t=sigma_t, s=s, plot_eval_metrics=False)
        quantization_error = som.quantization_error(data)
        silhouette = som.silhouette_score(data)
        db_index = som.davies_bouldin_score(data)
        num_clusters = som.get_number_of_clusters(data)
        results.append({
            'som': som,
            'neighbourhood_function': function,
            'quantization_error': quantization_error,
            'silhouette_score': silhouette,
            'davies_bouldin_index': db_index,
            'num_clusters': num_clusters
        })
    return results

#number of iterations
def try_different_number_of_iteration(data, M, N, input_dim, neighbourhood_function, 
                              number_of_iterations_values, lambda_, sigma_t, s):
    results = []
    for it in number_of_iterations_values:
        som=KohonenNetwork(M=M, N=N, input_dim=input_dim, neighbourhood_function=neighbourhood_function)
        som.fit(data, number_of_iterations=it, lambda_=lambda_, 
                sigma_t=sigma_t, s=s, plot_eval_metrics=False)
        quantization_error = som.quantization_error(data)
        silhouette = som.silhouette_score(data)
        db_index = som.davies_bouldin_score(data)
        num_clusters = som.get_number_of_clusters(data)
        results.append({
            'som': som,
            'number_of_iterations': it,
            'quantization_error': quantization_error,
            'silhouette_score': silhouette,
            'davies_bouldin_index': db_index, 
            'num_clusters': num_clusters
        })
    return results
        

#lambda
def try_different_lamda_values(data, M, N, input_dim, neighbourhood_function, 
                              number_of_iterations, lambda_values, sigma_t, s, 
                              plot_eval_metrics=True, eval_every=100):
    results = []
    for lambda_ in lambda_values:
        print(f"Fit with lambda value: {lambda_}")
        som=KohonenNetwork(M=M, N=N, input_dim=input_dim, neighbourhood_function=neighbourhood_function)
        som.fit(data, number_of_iterations=number_of_iterations, lambda_=lambda_, 
                sigma_t=sigma_t, s=s, plot_eval_metrics=plot_eval_metrics, eval_every=eval_every)
        quantization_error = som.quantization_error(data)
        silhouette = som.silhouette_score(data)
        db_index = som.davies_bouldin_score(data)
        num_clusters = som.get_number_of_clusters(data)
        results.append({
            'som': som,
            'lambda': lambda_,
            'quantization_error': quantization_error,
            'silhouette_score': silhouette,
            'davies_bouldin_index': db_index, 
            'num_clusters': num_clusters
        })
    return results

#sigma_t
def try_different_sigma_values(data, M, N, input_dim, neighbourhood_function, 
                              number_of_iterations, lambda_, sigma_t_values, s, 
                              plot_eval_metrics=True, eval_every=100):
    results = []
    for sigma in sigma_t_values:
        print(f"Fit with sigma value: {sigma}")
        som=KohonenNetwork(M=M, N=N, input_dim=input_dim, neighbourhood_function=neighbourhood_function)
        som.fit(data, number_of_iterations=number_of_iterations, lambda_=lambda_, 
                sigma_t=sigma, s=s, plot_eval_metrics=plot_eval_metrics, eval_every=eval_every)
        quantization_error = som.quantization_error(data)
        silhouette = som.silhouette_score(data)
        db_index = som.davies_bouldin_score(data)
        num_clusters = som.get_number_of_clusters(data)
        results.append({
            'som': som,
            'sigma': sigma,
            'quantization_error': quantization_error,
            'silhouette_score': silhouette,
            'davies_bouldin_index': db_index, 
            'num_clusters': num_clusters
        })
    return results


#trying different s_values from [0.1, 10] for neighbourhood function
def try_different_s_values(data, M, N, input_dim, neighbourhood_function, number_of_iterations, 
                           lambda_, sigma_t):
    s_values=[0.1, 0.5, 1.0, 2.0, 5.0, 10]
    results = []
    for s in s_values:
        print(f'Fit with s value: {s}')
        som=KohonenNetwork(M=M, N=N, input_dim=input_dim, neighbourhood_function=neighbourhood_function)
        som.fit(data, number_of_iterations=number_of_iterations, lambda_=lambda_, 
                sigma_t=sigma_t, s=s, plot_eval_metrics=False)
        quantization_error = som.quantization_error(data)
        silhouette = som.silhouette_score(data)
        db_index = som.davies_bouldin_score(data)
        num_clusters = som.get_number_of_clusters(data)
        results.append({
            'som': som,
            's': s,
            'quantization_error': quantization_error,
            'silhouette_score': silhouette,
            'davies_bouldin_index': db_index, 
            'num_clusters': num_clusters
        })
    return results
        
