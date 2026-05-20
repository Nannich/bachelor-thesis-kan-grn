import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import argparse
import math
import pandas as pd
import scanpy as sc

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




def plot_clusters(trajectory_cube, cluster_labels, lineage=None, n_bins=20, max_clusters_per_fig=5):
    """
    Plots the smoothed trajectories of genes grouped by their cluster assignments.
    """
    n_genes, n_lineages, _ = trajectory_cube.shape
    n_clusters = len(np.unique(cluster_labels))
    x_vals = np.linspace(0, 1, n_bins) # Scaled pseudotime axis for visualization

    if lineage is not None:
        # Calculate grid size (max 4 columns wide)
        cols = min(4, n_clusters)
        rows = math.ceil(n_clusters / cols)
        
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), sharex=True, sharey=True)
        axes = np.array(axes).flatten() 
            
        for c in range(n_clusters):
            ax = axes[c]
            gene_indices = np.where(cluster_labels == c)[0]
            cluster_data = trajectory_cube[gene_indices, lineage, :]

            # Plot individual gene lines
            for i in range(len(gene_indices)):
                ax.plot(x_vals, cluster_data[i], alpha=0.15, color='gray')

            # Plot cluster mean trend
            if len(gene_indices) > 0:
                ax.plot(x_vals, cluster_data.mean(axis=0), color='crimson', linewidth=2.5)
            
            ax.set_title(f"Cluster {c}\n(n={len(gene_indices)} genes)")
            if c >= len(axes) - cols: # Only add x-labels to the bottom row
                ax.set_xlabel("Pseudotime (Scaled)")
            if c % cols == 0:         # Only add y-labels to the first column
                ax.set_ylabel(f"Lineage {lineage} Expression")
                
        # Hide any unused subplots in the grid
        for c in range(n_clusters, len(axes)):
            axes[c].set_visible(False)
            
        plt.tight_layout()
        plt.show()
        
    else:
        # Break clusters into chunks
        cluster_chunks = [list(range(i, min(i + max_clusters_per_fig, n_clusters))) 
                          for i in range(0, n_clusters, max_clusters_per_fig)]
        
        for chunk_idx, chunk in enumerate(cluster_chunks):
            n_rows = len(chunk)
            
            fig, axes = plt.subplots(n_rows, n_lineages, 
                                     figsize=(4 * n_lineages, 3 * n_rows), 
                                     sharey='row', sharex=True)
            
            if n_rows == 1 and n_lineages == 1: axes = np.array([[axes]])
            elif n_rows == 1: axes = np.expand_dims(axes, 0)
            elif n_lineages == 1: axes = np.expand_dims(axes, 1)

            for row_idx, c in enumerate(chunk):
                gene_indices = np.where(cluster_labels == c)[0]
                
                for l in range(n_lineages):
                    ax = axes[row_idx, l]
                    cluster_data = trajectory_cube[gene_indices, l, :]

                    # Plot individual gene lines
                    for i in range(len(gene_indices)):
                        ax.plot(x_vals, cluster_data[i], alpha=0.15, color='gray')

                    # Plot cluster mean trend
                    if len(gene_indices) > 0:
                        ax.plot(x_vals, cluster_data.mean(axis=0), color='royalblue', linewidth=2.5)
                    
                    if row_idx == 0:
                        ax.set_title(f"Lineage {l+1} (Part {chunk_idx + 1})", fontweight='bold')
                    if l == 0:
                        ax.set_ylabel(f"Cluster {c}\n(n={len(gene_indices)})", fontweight='bold')
                    if row_idx == n_rows - 1:
                        ax.set_xlabel("Pseudotime (Scaled)")

            plt.tight_layout()
            #plt.show()


def plot_gene_vs_gene(adata, gene_x, gene_y):
    """
    Plots cell-by-cell expression of gene_x against gene_y in log1p space.
    """
    from src.utils import get_raw_counts
    
    gene_names = list(adata.var_names)
    
    if gene_x not in gene_names:
        raise ValueError(f"Gene '{gene_x}' not found in dataset var_names.")
    if gene_y not in gene_names:
        raise ValueError(f"Gene '{gene_y}' not found in dataset var_names.")
        
    idx_x = gene_names.index(gene_x)
    idx_y = gene_names.index(gene_y)
    
    # Extract counts and convert to log1p space matching model dimensions
    raw_counts = get_raw_counts(adata)
    x_val = np.log1p(raw_counts[:, idx_x])
    y_val = np.log1p(raw_counts[:, idx_y])
    
    plt.figure(figsize=(8, 6))
    
    plt.scatter(x_val, y_val, alpha=0.4, color='blue', s=15, edgecolors='none', label='Cells')
    
    plt.title(f"{gene_x} vs {gene_y}")
    plt.xlabel(f"Log({gene_x} + 1)")
    plt.ylabel(f"Log({gene_y} + 1)")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    
    out_dir = "figures/visualize/gene_vs_gene"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{gene_x}_vs_{gene_y}.png")
    plt.savefig(out_path, bbox_inches='tight', dpi=300)
    print(f" Saved gene vs gene plot to: {out_path}")
    plt.show()

from pathlib import Path

def plot_ground_truth_network(network_csv_path: str):
    """
    Plots the grn graph from the beeline edges list.
    """
    if not os.path.exists(network_csv_path):
        raise FileNotFoundError(f"Target network file missing: {network_csv_path}")

    df = pd.read_csv(network_csv_path)
    
    df.columns = [col.capitalize() for col in df.columns]

    G = nx.DiGraph()
    
    for _, row in df.iterrows():
        g1, g2, edge_type = str(row['Gene1']), str(row['Gene2']), str(row['Type'])
        G.add_edge(g1, g2, type=edge_type)

    edges = G.edges(data=True)
    colors = ['royalblue' if d['type'] == '+' else 'crimson' for u, v, d in edges]
    
    # Differentiate self-loops
    widths = [1.5 if u == v else 2.5 for u, v, d in edges]

    plt.figure(figsize=(10, 8))
    
    pos = nx.kamada_kawai_layout(G)
    
    nx.draw_networkx_nodes(G, pos, node_color='lightgray', node_size=800, edgecolors='white', linewidths=2)
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
    
    nx.draw_networkx_edges(
        G, pos, 
        edge_color=colors, 
        width=widths, 
        arrowsize=18, 
        connectionstyle='arc3,rad=0.15', 
        alpha=0.8
    )
    
    legend_handles = [
        mpatches.Patch(color='royalblue', label='Activation (+)'),
        mpatches.Patch(color='crimson', label='Repression (-)')
    ]
    plt.legend(handles=legend_handles, loc='upper right')
    plt.title(f"Ground Truth GRN Architecture: {os.path.basename(os.path.dirname(network_csv_path))}", fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    
    out_dir = "figures/visualize/ground_truth"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{os.path.basename(os.path.dirname(network_csv_path))}_gt_network.png")
    plt.savefig(out_path, bbox_inches='tight', dpi=300)
    print(f"Saved grn graph to: {out_path}")
    plt.show()


if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    
    parser = argparse.ArgumentParser()
    parser.add_argument("gene_x", type=str, nargs="?", default=None)
    parser.add_argument("gene_y", type=str, nargs="?", default=None)
    parser.add_argument("--dataset", type=str, default="li")
    parser.add_argument("--data_dir", type=str, default="./data/")
    
    args = parser.parse_args()
    base_search_path = Path(args.data_dir)
    
    matched_files = list(base_search_path.rglob(f"**/{args.dataset}/GroundTruthNetwork.csv"))
    if not matched_files:
        matched_files = list(base_search_path.rglob(f"**/{args.dataset.upper()}/GroundTruthNetwork.csv"))
        
    if not matched_files:
        print(f"Could not locate a GroundTruthNetwork.csv")
        sys.exit(1)
        
    target_network_path = str(matched_files[0])
    plot_ground_truth_network(target_network_path)

    """
    cache_file = os.path.join(args.data_dir, f"{args.dataset}_processed.h5ad")
    
    if not os.path.exists(cache_file):
        print(f"Preprocessing missing")
        sys.exit(1)
        
    adata = sc.read_h5ad(cache_file)
    
    plot_gene_vs_gene(adata, args.gene_x, args.gene_y)
    """