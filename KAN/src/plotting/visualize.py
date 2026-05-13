import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from src.utils import *
from src.formulas import *
from src.dataloaders import *
from src.model import build_model

KNOWN_GENES = {
    1252: "Gata1", 
    1670: "Klf1",  # Erythroid
    1913: "Mpo", 
    1040: "Elane", # Myeloid
    664:  "Cebpa", 
    619:  "Cd34",  # Progenitor
    1253: "Gata2"
}

def plot_parameters(ax, model, checkpoint, gene_to_plot):
    """
    Plots a textbox with the hyperparameters and global/local metrics of the model.
    """
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_type = checkpoint["model"]
    hidden_layers = checkpoint["hidden_layers"]
    lr = checkpoint["lr"]
    wd = checkpoint["wd"]
    gene = checkpoint["gene"]
    
    # Metrics
    mse = checkpoint["mse"]          # Shape: (n_genes, n_lineages)
    aic = checkpoint["aic"]          # Shape: (n_genes,)
    bic = checkpoint["bic"]          # Shape: (n_genes,)
    zinb_loss = checkpoint.get("zinb_loss", 0.0)
    global_aic = checkpoint.get("global_aic", 0.0)
    global_bic = checkpoint.get("global_bic", 0.0)
    
    total_avg_mse = mse.mean().item()

    gene_idx = 0 if mse.shape[0] == 1 else gene_to_plot
    
    mse_lineages = [f"L{l+1}: {mse[gene_idx, l].item():.3f}" for l in range(mse.shape[1])]
    mse_str = " | ".join(mse_lineages)
    
    aic_val = aic[gene_idx].item()
    bic_val = bic[gene_idx].item()

    # Construct the text box
    text = (
        f"Model: {model_type.upper()}\n"
        f"Layers: {hidden_layers}\n"
        f"Params: {total_params:,}\n"
        f"-------------------\n"
        f"ZINB Loss: {zinb_loss:.4f}\n"
        f"Avg MSE: {total_avg_mse:.4f}\n"
        f"Global AIC: {global_aic:,.0f}\n"
        f"Global BIC: {global_bic:,.0f}\n"
        f"-------------------\n"
        f"MSE: {mse_str}\n"
        f"AIC: {aic_val:.0f} | BIC: {bic_val:.0f}"
    )

    text_box = ax.text(1.02, 0.5, text, transform=ax.transAxes, 
            fontsize=8.5, verticalalignment='center', horizontalalignment='left',
            linespacing=1.4,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))

    return text_box

def plot_curves(ax, pseudotime, weights, model, gene_to_plot, checkpoint, colors):
    """
    Plots the raw and smoothed prediction curves of the model.
    """

    model.eval()

    gene = checkpoint["gene"]
    pt_min = checkpoint["pt_min"]
    pt_max = checkpoint["pt_max"]

    is_single_gene = gene is not None
    
    n_lineages = weights.shape[1]

    predictions_raw = predict_lineage_trajectories(pseudotime, weights, model, gene_to_plot, pt_min, pt_max)

    for l in range(n_lineages):
        x_raw, _, y_raw = predictions_raw[l]
        x_smooth, y_smooth = smoothen_lineage_trajectory(x_raw, y_raw)

        #ax.plot(x_raw, y_raw, linewidth=1, color=colors[l], label=f"Raw Predictions - Lineage {l+1}", alpha=0.4)
        ax.plot(x_smooth, y_smooth, linewidth=3, color=colors[l], label=f"Smoothed - Lineage {l+1}", alpha=0.6)
        

def plot_custom(ax, pseudotime, checkpoint, colors):
    """
    Plots the curve of a custom formula using a clean linspace.
    """
    pt_min = checkpoint["pt_min"]
    pt_max = checkpoint["pt_max"]

    n_lineages = pseudotime.shape[1]
    for l in range(n_lineages):
        # Create sorted X-axis from 0 to the max pseudotime of this lineage
        max_pt_for_lineage = np.max(pseudotime[:, l])
        x_clean = np.linspace(0, max_pt_for_lineage, 300)
        
        # Scale the x axis
        pt_input_scaled = scale_pt(x_clean, pt_min, pt_max)
        y_formula_raw = pykan_paul_gene1670(pt_input_scaled, lineage=l)
        
        # Transform back to log1p space and flatten to 1D
        y_formula = np.log1p(np.exp(y_formula_raw)).flatten()

        
        ax.plot(x_clean, y_formula, linewidth=4, color=colors[l], linestyle="--", label=f"Lineage {l+1} (Symbolic)", zorder=4)

def plot_scatter_data(ax, adata, pseudotime, weights, gene_to_plot, colors):
    """
    Plots the expression count of each cell for each lineage it is in.
    """
    lineage_assignment = get_lineage_assignment(weights)
    n_lineages = lineage_assignment.shape[1]
    
    # Get raw counts for plotting 
    raw_counts = get_raw_counts(adata)
        
    for l in range(n_lineages):
        mask = lineage_assignment[:, l]
        pt_active = pseudotime[mask, l]
        log_count_active = np.log1p(raw_counts[mask, gene_to_plot])
        ax.scatter(pt_active, log_count_active, s=16, color=colors[l], alpha=0.4)


def plot_everything(adata, pseudotime, weights, model, checkpoint, gene_to_plot, fig_path):
    """
    Plots the scatter data and curves.
    """
    model.eval()

    n_lineages = weights.shape[1]    
    colors = plt.get_cmap('viridis')(np.linspace(0, 1, n_lineages))
    # colors = ['#084594', '#d95f02']
    # colors = ['#000000', '#000000']


    fig, ax = plt.subplots(figsize=(10, 6))

    text_box = plot_parameters(ax, model, checkpoint, gene_to_plot,)
    
    plot_scatter_data(ax, adata, pseudotime, weights, gene_to_plot, colors)
    
    plot_curves(ax, pseudotime, weights, model, gene_to_plot, checkpoint, colors)
    plot_custom(ax, pseudotime, checkpoint, colors)
    
    ax.set_title(f"Gene: {gene_to_plot}")

    gene_name = KNOWN_GENES.get(gene_to_plot, f"Gene {gene_to_plot}")
    ax.set_title(f"{gene_name} Expression Trajectory", fontsize=16, fontweight='bold')

    ax.set_xlabel("Pseudotime")
    ax.set_ylabel("Log(expression + 1)")
    ax.legend(title="Lineage")

    fig.subplots_adjust(left=0.05, bottom=0.08, top=0.92, right=0.8)

    plt.savefig(fig_path, bbox_inches="tight", dpi=300)
    plt.savefig(fig_path, bbox_inches="tight", bbox_extra_artists=(text_box,), dpi=300)
    plt.show()


def run_visualization(args, adata, pseudotime, weights):
    gene_to_plot = args.gene
    model_dir = args.model_dir
    fig_dir = args.fig_dir
    model_name = args.name
    dataset = args.dataset

    model_path = os.path.join(model_dir, model_name)

    checkpoint = torch.load(model_path, weights_only=False)
    
    model_type = checkpoint["model"]
    input_dim = checkpoint["input_dim"]
    output_dim = checkpoint["output_dim"]

    fig_path = os.path.join(fig_dir, "visualize", dataset, f"{model_type}_{dataset}_gene{gene_to_plot}.png")
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    model = build_model(model_type, input_dim, output_dim)
    model.load_state_dict(checkpoint["state_dict"])

    plot_everything(adata, pseudotime, weights, model, checkpoint, gene_to_plot, fig_path)

