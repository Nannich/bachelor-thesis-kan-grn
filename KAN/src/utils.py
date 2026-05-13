import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from efficient_kan import KAN

def get_lineage_assignment(weights):
    """
    Assigns the cells to lineages based on their weight.
    A cell can be part of two lineages at the same time, e.g. if it's weight is [1, 1].
    """
    sensitivity = 0.1
    max_weights = np.max(weights, axis=1, keepdims=True)
    lineage_assignment = np.abs(max_weights - weights) < sensitivity

    return lineage_assignment


def scale_pt(pt, pt_min, pt_max):
    return (pt - pt_min) / (pt_max - pt_min + 1e-8)


def sort_by_lineage(pseudotime, weights, lineage):
    """
    Extracts and sorts the active pseudotime and weights for a specific lineage.
    Returns the pseudotimes and weights of each lineage sorted by the given lineage's pseudtime.
    """
    lineage_assignment = get_lineage_assignment(weights)
    mask = lineage_assignment[:, lineage]

    pt_active = pseudotime[mask]
    weights_active = weights[mask]

    # Get sort indices based on the target lineage's column
    sort_idx = np.argsort(pt_active[:, lineage])
    
    # Apply sort to 2D matrices
    pt_sorted = pt_active[sort_idx]
    weights_sorted = weights_active[sort_idx]
    
    return pt_sorted, weights_sorted

def filter_predictions(predictions, is_de):
    """
    Filters the predictions in the dictionary by gene. 
    """
    filtered = {}
    for l, (pt, scaled, y_raw) in predictions.items():
        y_de = y_raw[:, is_de] if y_raw.ndim > 1 else y_raw
        filtered[l] = (pt, scaled, y_de)
    return filtered

def predict_lineage_trajectories(pseudotime, weights, model, gene_idx, pt_min, pt_max):
    """
    Runs the model once on each lineage and returns the predicted values in a dictionary 
    (because lineages might differ in length).
    """
    n_lineages = weights.shape[1]
    predictions = {}
    
    model.eval()

    for lineage in range(n_lineages):
        pt_sorted, weights_sorted = sort_by_lineage(pseudotime, weights, lineage)
        pt_sorted_active = pt_sorted[:, lineage]

        # Scale and prepare inputs
        pt_input_scaled = scale_pt(pt_sorted, pt_min, pt_max)
        input_matrix = np.hstack((pt_input_scaled, weights_sorted))
        X_tensor = torch.tensor(input_matrix, dtype=torch.float32)

        # Run model
        with torch.no_grad():
            mu, theta, pi = model(X_tensor)

        mu_np = mu.detach().cpu().numpy()

        if gene_idx is None:
            y_line = mu_np  # Keep all genes, shape is (n_cells, n_genes)
        else:
            # Determine the integer index
            idx = 0 if mu_np.shape[1] == 1 else gene_idx
            # Forces the output to stay 2D: (n_cells, 1)
            y_line = mu_np[:, [idx]]
        
        # Model predicts log counts but predictions should be log1p
        y_line = np.exp(y_line)                         
        y_line = np.log1p(y_line)

        # Store the results for this lineage in the dictionary
        predictions[lineage] = (pt_sorted_active, pt_input_scaled, y_line)

    return predictions


def smoothen_lineage_trajectory(pseudotime, y_line, n_bins=20):
    """
    Smoothens the predictions by averaging the predicted counts into n_bin intervals.
    """

    # Create equally spaced bin boundaries from 0 to the maximum pseudotime
    bin_edges = np.linspace(0, np.max(pseudotime), n_bins + 1)

    # Calculate the center of each bin to use as the new x-axis coordinates
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Assign each sample's pseudotime to a bin index
    bin_indices = np.digitize(pseudotime, bin_edges) - 1

    # Edge case: index is the exact max value
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    n_genes = y_line.shape[1]
    y_smoothed = np.full((n_bins, n_genes), np.nan)

    # Iterate through each bin and calculate the mean expression of all samples in it
    for i in range(n_bins):
        mask = (bin_indices == i)
        if np.any(mask):
            y_smoothed[i, :] = np.mean(y_line[mask, :], axis=0)

    # Handles empty bins:
    # - Linear interpolation fills gaps between two bins
    # - ffill/bfill handles cases where the first or last bins are empty
    df_smoothed = pd.DataFrame(y_smoothed)
    y_final = df_smoothed.interpolate(method='linear', axis=0).ffill().bfill().values

    # Return the new x-coordinates and the y-values
    return bin_centers, y_final


def build_smoothed_cube(filtered_predictions, n_bins=20):
    """
    Converts a dictionary of varying-length lineages into a 
    fixed-size 3D Matrix: (Gene, Lineage, Bin).
    """
    n_lineages = len(filtered_predictions)
    # Grab first lineage to see how many DE genes we have
    _, _, first_y = filtered_predictions[0]
    n_de_genes = first_y.shape[1]

    # Shape: (DE Genes, Lineages, Bins)
    trajectory_matrix = np.zeros((n_de_genes, n_lineages, n_bins))

    for l, (pt, _, y_de) in filtered_predictions.items():
        # Smoothening creates a fixed 'n_bins' length
        _, y_smooth = smoothen_lineage_trajectory(pt, y_de, n_bins=n_bins)
        
        # y_smooth is (n_bins, n_de_genes), so we transpose to (n_de_genes, n_bins)
        trajectory_matrix[:, l, :] = y_smooth.T

    return trajectory_matrix


def get_raw_counts(adata):
    # Use .raw if it exists, otherwise fall back to .X
    data_source = adata.raw.X if adata.raw is not None else adata.X
    
    # Check if it has the 'toarray' method (indicates it's a Scipy sparse matrix)
    if hasattr(data_source, "toarray"):
        return data_source.toarray().astype(np.float32)
    
    # Otherwise, assume it's already a numpy array or similar
    return np.array(data_source, dtype=np.float32)