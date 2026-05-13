import numpy as np
import os
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from kan import KAN as PyKAN

from src.utils import *
from src.analysis.de import *
from src.model import build_model
from src.plotting.plot import *

N_BINS = 512

def train_n_to_1_kan(X_tensor, Y_tensor, epochs=150, lr=0.01, lamb_l1=0.1):
    in_dim = X_tensor.shape[1]
    model = PyKAN(width=[in_dim, 1], grid=3, k=3, device="cpu")
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        predictions = model(X_tensor)
        
        mse_loss = criterion(predictions, Y_tensor)
        
        # Apply L1 Penalty for sparse GRN
        l1_loss = sum(torch.sum(torch.abs(param)) for param in model.parameters())
        
        loss = mse_loss + (lamb_l1 * l1_loss)
        loss.backward()
        optimizer.step()
        
    model.eval()
    with torch.no_grad():
        final_predictions = model(X_tensor)
        
    return model, final_predictions.detach().numpy()

def get_correlation_signs(X_numpy, Y_numpy):
    # Determines if an edge is an activator or repressor
    signs = np.zeros(X_numpy.shape[1])
    y_flat = Y_numpy.flatten()
    
    for i in range(X_numpy.shape[1]):
        x_col = X_numpy[:, i]
        
        if np.std(x_col) == 0 or np.std(y_flat) == 0:
            corr = 0
        else:
            corr = np.corrcoef(x_col, y_flat)[0, 1]
            
        signs[i] = 1 if corr >= 0 else -1
        
    return signs


def evaluate_grn(adj_matrix, gene_names, dataset_name, data_dir="./data", edge_threshold=0.02):
    # Evaluates the inferred GRN against the ground truth network.
    gt_path = os.path.join(data_dir, dataset_name, "GroundTruthNetwork.csv")
    out_path = os.path.join(data_dir, f"grn_{dataset_name}.csv") 

    # Load Ground Truth
    gt_df = pd.read_csv(gt_path)
    gene_list = list(gene_names)
    n_genes = len(gene_list)
    
    # Store true edges: keys are (Gene1, Gene2), values are sign (+1 or -1)
    true_edges = {}
    inferred_weights = []
    
    # For AUROC / AUPRC
    true_adj = np.zeros((n_genes, n_genes))
    
    for _, row in gt_df.iterrows():
        g1, g2 = str(row['Gene1']).strip(), str(row['Gene2']).strip()
        edge_type = str(row['Type']).strip()
        sign = 1 if edge_type == '+' else -1
        
        true_edges[(g1, g2)] = sign
        
        if g1 in gene_list and g2 in gene_list:
            idx1, idx2 = gene_list.index(g1), gene_list.index(g2)
            weight = adj_matrix[idx1, idx2]
            true_adj[idx1, idx2] = 1 # Mark edge existence for AUC
        else:
            weight = 0.0
            
        inferred_weights.append(weight)
        
    gt_df['Inferred_Weight'] = inferred_weights
    gt_df.to_csv(out_path, index=False)    


    y_true = true_adj.flatten()
    y_score = np.abs(adj_matrix).flatten()
    
    if len(np.unique(y_true)) > 1:
        auroc = roc_auc_score(y_true, y_score)
        auprc = average_precision_score(y_true, y_score)
    else:
        auroc, auprc = float('nan'), float('nan')

    exact_matches = 0
    sign_flips = 0
    dir_flips = 0
    dir_sign_flips = 0
    pure_fps = 0
    
    predicted_edges = {}
    for i, g1 in enumerate(gene_list):
        for j, g2 in enumerate(gene_list):
            w = adj_matrix[i, j]
            if abs(w) > edge_threshold:
                pred_sign = 1 if w > 0 else -1
                predicted_edges[(g1, g2)] = pred_sign
                
                if (g1, g2) in true_edges:
                    if true_edges[(g1, g2)] == pred_sign:
                        exact_matches += 1
                    else:
                        sign_flips += 1
                else:
                    if (g2, g1) in true_edges:
                        if true_edges[(g2, g1)] == pred_sign:
                            dir_flips += 1
                        else:
                            dir_sign_flips += 1
                    else:
                        pure_fps += 1
                
    n_predicted = len(predicted_edges)
    n_true = len(true_edges)
    fns = n_true - exact_matches - sign_flips # Edges missed entirely in forward direction
    
    print(f"Total True Edges Evaluated:       {n_true}")
    print(f"Total Predicted Edges:            {n_predicted} (Threshold > {edge_threshold})")
    print("-" * 40)
    print(f"AUROC:                            {auroc:.4f}")
    print(f"AUPRC:                            {auprc:.4f}")
    print("-" * 40)
    print("Prediction Breakdown:")
    print(f"  True Positives (Exact Matches): {exact_matches}")
    print(f"  Sign Flips (A->B, wrong sign):  {sign_flips}")
    print(f"  Dir Flips (B->A, same sign):    {dir_flips}")
    print(f"  Dir & Sign Flips (B->A, wrong): {dir_sign_flips}")
    print(f"  Pure False Positives:           {pure_fps}")
    print("-" * 40)
    print(f"False Negatives (Missed Edges):   {fns}")
    print("="*40 + "\n")


def get_grn(predictions):
    n_genes, _, _ = predictions.shape
    flattened_matrix = predictions.reshape(n_genes, -1).T 
    adj_matrix = np.zeros((n_genes, n_genes))
    
    for target_idx in range(n_genes):
        print(f"Evaluating Gene {target_idx+1}/{n_genes}")
        
        Y_numpy = flattened_matrix[:, [target_idx]]
        X_numpy = np.delete(flattened_matrix, target_idx, axis=1) # Drop the target gene from inputs
        
        X_tensor = torch.tensor(X_numpy, dtype=torch.float32)
        Y_tensor = torch.tensor(Y_numpy, dtype=torch.float32)
        
        # Train model with L1 penalty
        kan_model, predictions = train_n_to_1_kan(X_tensor, Y_tensor, epochs=150, lr=0.01, lamb_l1=0.02)
        
        # Extract Spline Magnitudes
        kan_model.attribute()
        edge_magnitudes = kan_model.edge_scores[0].detach().cpu().numpy().flatten()

        # Extract Correlation Signs
        edge_signs = get_correlation_signs(X_numpy, Y_numpy)
        
        # Calculate Final Directed Weights
        edge_weights = edge_magnitudes * edge_signs
        
        weights_full = np.insert(edge_weights, target_idx, 0.0) 
        adj_matrix[:, target_idx] = weights_full
    
    return adj_matrix

def save_beeline_ranked_edges(adj_matrix, gene_names, data_dir, dataset_name):
    """
    Converts your adjacency matrix into BEELINE's expected rankedEdges.csv format.
    BEELINE ranks edges purely by the absolute magnitude of the weight.
    """
    edges = []
    n_genes = len(gene_names)
    
    for i in range(n_genes):
        for j in range(n_genes):
            if i != j:  # Exclude self-loops
                weight = adj_matrix[i, j]
                abs_weight = abs(weight)
                if abs_weight > 0:
                    edges.append({
                        'Gene1': gene_names[i], # Source / Regulator
                        'Gene2': gene_names[j], # Target
                        'EdgeWeight': abs_weight, # BEELINE ranks by magnitude
                        'Sign': 1 if weight > 0 else -1 # Keeps track of activator/repressor
                    })
                    
    df = pd.DataFrame(edges)
    if not df.empty:
        df = df.sort_values(by='EdgeWeight', ascending=False)
        
    # Extract dataset group (e.g. 'GSD' or 'dyn-BF-100') from data_dir
    dataset_group = os.path.basename(os.path.normpath(data_dir))
    
    # Save exactly where BEELINE expects it
    out_dir = f"external/Beeline/outputs/{dataset_group}/{dataset_name}/myKAN"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "rankedEdges.csv")
    
    df.to_csv(out_path, sep='\t', index=False)
    print(f"Saved ranked edges for BEELINE to: {out_path}")

def run_grn(args, adata, pseudotime, weights):
    data_dir = args.data_dir
    model_dir = args.model_dir
    model_name = args.name
    dataset = args.dataset

    model_path = os.path.join(model_dir, model_name)
    checkpoint = torch.load(model_path, weights_only=False)
    pt_min = checkpoint["pt_min"]
    pt_max = checkpoint["pt_max"]

    null_name = f"null_{dataset}_all.pth"
    null_path = os.path.join(model_dir, null_name)
    #null_checkpoint = torch.load(null_path, weights_only=False)
    
    # Rebuild original model to extract DE trajectories
    model_type = checkpoint["model"]
    input_dim = checkpoint["input_dim"]
    output_dim = checkpoint["output_dim"]
    
    model = build_model(model_type, input_dim, output_dim)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    predictions = predict_lineage_trajectories(pseudotime, weights, model, None, pt_min, pt_max)

    n_genes = output_dim // 3
    is_de = np.ones(n_genes, dtype=bool)
    # is_de = information_criteria_test(checkpoint, null_checkpoint, threshold=-99999)

    predictions_de = filter_predictions(predictions, is_de)

    predictions_smooth = build_smoothed_cube(predictions_de, n_bins=N_BINS)

    de_gene_names = adata.var_names.values[is_de]
        
    adj_matrix = get_grn(predictions_smooth)

    adj_df = pd.DataFrame(adj_matrix, index=de_gene_names, columns=de_gene_names)
    output_filename = f"results/{model_name}_grn.csv"
    adj_df.to_csv(output_filename)

    #evaluate_grn(adj_matrix, de_gene_names, dataset, data_dir=args.data_dir, edge_threshold=0.01)
    
    plot_grn(adj_matrix, de_gene_names, edge_threshold=0.3)

    save_beeline_ranked_edges(adj_matrix, de_gene_names, args.data_dir, args.dataset)