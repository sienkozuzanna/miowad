import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import pandas as pd

def visualize_2d_data(data, figsize=(6,4), title="2D Data Visualization", xlabel="X-axis", ylabel="Y-axis"):
    plt.figure(figsize=figsize)
    plt.scatter(data[:, 0], data[:, 1], c='blue', marker='o')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    plt.show()

def visualize_3d_data(data, title="3D Data Visualization", 
                        labels=['X-axis', 'Y-axis', 'Z-axis'], point_size=3, opacity=0.8):
    fig = go.Figure(data=[go.Scatter3d(x=data[:, 0],y=data[:, 1],z=data[:, 2],
        mode='markers',marker=dict(
            size=point_size,
            color='blue',
            opacity=opacity
        )
    )])
    
    fig.update_layout(title=title,
        scene=dict(
            xaxis_title=labels[0],
            yaxis_title=labels[1],
            zaxis_title=labels[2]
        ), margin=dict(l=0, r=0, b=0, t=30))
    fig.show()


def print_neuron_mapping(class_to_neurons, neuron_stats):
    stats_data = []
    for (i,j), (cls, purity) in neuron_stats.items():
        stats_data.append({
            'Neuron': f"({i},{j})",
            'Dominant Class': cls,
            'Purity': f"{purity:.1%}",
            'Associated Classes': [k for k,v in class_to_neurons.items() if (i,j) in v]
        })

    df = pd.DataFrame(stats_data)
    df.sort_values(['Dominant Class', 'Purity'], ascending=[True, False], inplace=True)
    
    print("="*60)
    print("NEURON CLASS MAPPING REPORT".center(60))
    print("="*60)
    print(df.to_string(index=False))
    print("\n" + "CLASS TO NEURON MAPPING SUMMARY".center(60))
    print("-"*60)
    for cls, neurons in class_to_neurons.items():
        print(f"Class {cls}: {len(neurons)} neurons → {sorted(neurons)}")

def print_class_distribution(class_dist):
    stats_data = []
    for (i,j), counts in class_dist.items():
        total = sum(counts.values()) if counts else 0
        dist_str = ', '.join(f"{c}:{cnt}({cnt/total:.1%})" 
                            for c, cnt in sorted(counts.items())) if counts else 'Empty'
        
        stats_data.append({
            'Neuron': f"({i},{j})",
            'Total Samples': total,
            'Class Distribution': dist_str
        })
    
    df = pd.DataFrame(stats_data)
    print("="*80)
    print("CLASS DISTRIBUTION IN NEURONS".center(80))
    print("="*80)
    print(df.to_string(index=False, justify='center'))