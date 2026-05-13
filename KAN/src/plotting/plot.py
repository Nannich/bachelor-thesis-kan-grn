import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from src.utils import *
from src.analysis.de import *
from src.model import build_model

def plot_grn(adj_matrix, gene_names, edge_threshold=0.2):
    G = nx.DiGraph()
    n_genes = adj_matrix.shape[0]
    
    for i in range(n_genes):
        G.add_node(i, label=gene_names[i])
        
    # Add edges that survive above the threshold
    for i in range(n_genes):
        for j in range(n_genes):
            weight = adj_matrix[i, j]
            if abs(weight) > edge_threshold:
                G.add_edge(i, j, weight=weight)
                
    edges = G.edges(data=True)
    colors = ['royalblue' if d['weight'] > 0 else 'crimson' for u, v, d in edges]
    
    # Scale widths
    max_weight = np.max(np.abs(adj_matrix))
    widths = [(abs(d['weight']) / max_weight) * 5 for u, v, d in edges]
    
    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(G, k=2.0) 
    
    nx.draw_networkx_nodes(G, pos, node_color='lightgray', node_size=600, edgecolors='white', linewidths=2)
    nx.draw_networkx_edges(G, pos, edge_color=colors, width=widths, arrowsize=15, connectionstyle='arc3,rad=0.1', alpha=0.7)
    
    labels = {i: gene_names[i] for i in range(n_genes)}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=10, font_weight='bold')
    #labels = {i: str(i) for i in range(n_genes)}
    #nx.draw_networkx_labels(G, pos, labels=labels, font_size=10, font_weight='bold')
    
    legend_handles = [
        mpatches.Patch(color='royalblue', label='Activation (+)'),
        mpatches.Patch(color='crimson', label='Repression (-)')
    ]
    plt.legend(handles=legend_handles, loc='upper right')
    
    plt.axis('off')
    plt.tight_layout()
    plt.show()