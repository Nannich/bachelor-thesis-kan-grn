import numpy as np
import os
from pathlib import Path
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
from src.train import run_training



USE_SMOOTH = False
TRAJECTORY_MODEL_TYPE = "effkan"
N_BINS = 512

def train_n_to_1_kan(X_tensor, Y_tensor, epochs=600, lr=0.01, lamb_l1=0.1):
    in_dim = X_tensor.shape[1]
    model = PyKAN(
        width=[in_dim, 1], 
        grid=3, 
        k=3,
        device="cpu", 
        auto_save=False
    )
    model.update_grid_from_samples(X_tensor)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        predictions = model(X_tensor)
        mse_loss = criterion(predictions, Y_tensor)
        
        l1_loss = sum(torch.sum(torch.abs(param)) for param in model.parameters())
                
        loss = mse_loss + (lamb_l1 * l1_loss)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        final_predictions = model(X_tensor)
        
    return model, final_predictions.detach().numpy()


def get_correlation_signs(X_numpy, Y_numpy):
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


def train_grn_models(expression_matrix, gene_names, save_dir, is_smoothed=False):
    """ Trains and saves a self-contained checkpoint for each gene. """
    os.makedirs(save_dir, exist_ok=True)
    n_samples, n_genes = expression_matrix.shape

    epochs = 600
    #lamb_l1 = 0.05 if is_smoothed else 0.02
    lamb_l1 = 0.02

    for target_idx in range(n_genes):
        target_gene = gene_names[target_idx]
        model_save_path = os.path.join(save_dir, f"{target_gene}_kan.pth")
                
        print(f"Training Gene {target_idx+1}/{n_genes}: {target_gene} (Smooth Mode: {is_smoothed})")
        
        Y_numpy = expression_matrix[:, [target_idx]]
        X_numpy = np.delete(expression_matrix, target_idx, axis=1)
        X_tensor, Y_tensor = torch.tensor(X_numpy, dtype=torch.float32), torch.tensor(Y_numpy, dtype=torch.float32)
        
        kan_model, _ = train_n_to_1_kan(X_tensor, Y_tensor, lr=0.01, lamb_l1=lamb_l1, epochs=epochs)
        
        checkpoint = {
            "state_dict": kan_model.state_dict(),
            "X_numpy": X_numpy,
            "Y_numpy": Y_numpy,
            "grid": kan_model.grid,
            "k": kan_model.k
        }
        torch.save(checkpoint, model_save_path)


def extract_grn_matrix(gene_names, save_dir):
    """ Extracts adjacency matrix directly from saved checkpoints. """
    n_genes = len(gene_names)
    adj_matrix = np.zeros((n_genes, n_genes))

    for target_idx in range(n_genes):
        target_gene = gene_names[target_idx]
        model_save_path = os.path.join(save_dir, f"{target_gene}_kan.pth")
        
        checkpoint = torch.load(model_save_path, weights_only=False)
        grid_size = checkpoint.get("grid", 3)
        k_order = checkpoint.get("k", 3)
        
        kan_model = PyKAN(width=[n_genes - 1, 1], grid=grid_size, k=k_order, device="cpu", auto_save=False)
        kan_model.load_state_dict(checkpoint["state_dict"])
        
        X_numpy = checkpoint["X_numpy"]
        Y_numpy = checkpoint["Y_numpy"]
        
        kan_model.eval()
        X_tensor = torch.tensor(X_numpy, dtype=torch.float32)
        with torch.no_grad():
            kan_model(X_tensor)
        
        kan_model.attribute()
        edge_magnitudes = kan_model.edge_scores[0].detach().cpu().numpy().flatten()
        edge_signs = get_correlation_signs(X_numpy, Y_numpy)
        
        edge_weights = edge_magnitudes * edge_signs
        adj_matrix[:, target_idx] = np.insert(edge_weights, target_idx, 0.0)
    
    return adj_matrix


def save_beeline_ranked_edges(adj_matrix, gene_names, data_dir, dataset_name, base_model_name):
    edges = []
    n_genes = len(gene_names)
    
    for i in range(n_genes):
        for j in range(n_genes):
            if i != j:
                weight = adj_matrix[i, j]
                abs_weight = abs(weight)
                if abs_weight > 0:
                    edges.append({
                        'Gene1': gene_names[i],
                        'Gene2': gene_names[j],
                        'EdgeWeight': abs_weight,
                        'Sign': 1 if weight > 0 else -1
                    })
                    
    df = pd.DataFrame(edges)
    if not df.empty:
        df = df.sort_values(by='EdgeWeight', ascending=False)
        
    path_dir = Path(data_dir)
    if "Synthetic" in path_dir.parts:
        base_path = Path("data/BEELINE-data/inputs/Synthetic")
        dataset_group = str(path_dir.relative_to(base_path))
    else:
        base_path = Path("data/BEELINE-data/inputs/Curated")
        dataset_group = str(path_dir.relative_to(base_path))
        
    out_dir = f"external/Beeline/outputs/{dataset_group}/{dataset_name}/{base_model_name}"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "rankedEdges.csv")
    
    df.to_csv(out_path, sep='\t', index=False)
    print(f"Saved ranked edges for BEELINE to: {out_path}")


def run_grn(args, adata, pseudotime, weights):
    dataset = args.dataset
    base_model_name = args.name if args.name else "temp"
    
    if USE_SMOOTH:
        
        print("Smooth")

        # Train a new trajetory prediction model
        class TrainArgsNamespace:
            model = TRAJECTORY_MODEL_TYPE
            model_dir = args.model_dir
            gene = None # Train tracking all genes simultaneously
            dataset = args.dataset
            
        run_training(TrainArgsNamespace(), adata, pseudotime, weights)
        
        # Use the new checkpoint file
        generated_model_name = f"{TRAJECTORY_MODEL_TYPE}_{dataset}_all.pth"
        model_path = os.path.join(args.model_dir, generated_model_name)
        
        checkpoint = torch.load(model_path, weights_only=False)
        pt_min, pt_max = checkpoint["pt_min"], checkpoint["pt_max"]
        model_type, input_dim, output_dim = checkpoint["model"], checkpoint["input_dim"], checkpoint["output_dim"]
        
        model = build_model(model_type, input_dim, output_dim)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()

        predictions = predict_lineage_trajectories(pseudotime, weights, model, None, pt_min, pt_max)
        n_genes = output_dim // 3
        is_de = np.ones(n_genes, dtype=bool)
        predictions_de = filter_predictions(predictions, is_de)
        
        predictions_smooth = build_smoothed_cube(predictions_de, n_bins=N_BINS)
        expression_matrix = predictions_smooth.reshape(n_genes, -1).T 

    else:
        print("Raw")
        raw_counts = np.log1p(get_raw_counts(adata))
        lineage_assignment = get_lineage_assignment(weights)
        n_lineages = weights.shape[1]
        n_genes = raw_counts.shape[1]
        is_de = np.ones(n_genes, dtype=bool)
        
        raw_matrices = []
        for l in range(n_lineages):
            mask = lineage_assignment[:, l]
            if not np.any(mask): continue
                
            pt_active = pseudotime[mask, l]
            counts_active = raw_counts[mask]
            
            sort_idx = np.argsort(pt_active)
            raw_matrices.append(counts_active[sort_idx])
            
        expression_matrix = np.vstack(raw_matrices)

    save_dir = os.path.join(args.model_dir, "grn_models", dataset, base_model_name)
    de_gene_names = adata.var_names.values[is_de]
    
    train_grn_models(expression_matrix, de_gene_names, save_dir, is_smoothed=USE_SMOOTH)
    adj_matrix = extract_grn_matrix(de_gene_names, save_dir)

    dataset_group = os.path.basename(os.path.normpath(args.data_dir))
    run_output_dir = f"results/grn/{base_model_name}"
    os.makedirs(run_output_dir, exist_ok=True)
    
    adj_df = pd.DataFrame(adj_matrix, index=de_gene_names, columns=de_gene_names)
    output_filename = f"{run_output_dir}/{dataset_group}_{dataset}_grn.csv"
    adj_df.to_csv(output_filename)

    save_beeline_ranked_edges(adj_matrix, de_gene_names, args.data_dir, args.dataset, base_model_name)