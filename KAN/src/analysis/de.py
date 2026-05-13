import numpy as np
import os
import torch
import torch.nn.functional as F

from src.utils import *
from src.model import build_model

def information_criteria_test(model_checkpoint, null_checkpoint, threshold=2.0):
    # Extract the pre-calculated AIC metrics
    model_aic = model_checkpoint["aic"].numpy()
    null_aic = null_checkpoint["aic"].numpy()

    delta_aic = null_aic - model_aic
    
    is_de = delta_aic > threshold

    return is_de

def association_test(model, pseudotime, weights, model_gene, threshold, pt_min, pt_max):
    n_lineages = pseudotime.shape[1]
    is_single_gene = model_gene is not None
    
    de_across_lineages = []
    
    predictions = predict_lineage_trajectories(pseudotime, weights, model, None, pt_min, pt_max)
    

    for lineage in range(n_lineages):
        pt_active_sorted, pt_input_scaled, y_pred = predictions[lineage]
        differences = np.max(y_pred, axis=0) - np.min(y_pred, axis=0)
        de_across_lineages.append(differences > threshold)
        
    return np.any(de_across_lineages, axis=0)


def evaluate(pred_de, true_de):
    n_true_de = np.sum(true_de)
    n_pred_de = np.sum(pred_de)

    tp_counts = np.sum(pred_de & true_de)
    tpr = tp_counts / n_true_de

    fd_counts = np.sum(pred_de & ~true_de)
    fdr = fd_counts / n_pred_de

    return tpr, fdr


def calculate_mse_per_curve(dataloader, model, device="cpu"):
    model.eval()
    
    total_squared_error = None
    samples_per_lineage = None

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            n_lineages = X.shape[1] // 2
            weights = X[:, n_lineages:]

            max_weights, _ = torch.max(weights, dim=1, keepdim=True)
            sensitivity = 0.1
            mask = torch.abs(max_weights - weights) < sensitivity  # Shape: (batch_size, n_lineages)

            mu, theta, pi = model(X)
            y_true_log1p = torch.log1p(y)
            y_pred_log1p = torch.log1p(torch.exp(mu))

            sq_err = F.mse_loss(y_pred_log1p, y_true_log1p, reduction='none')

            if total_squared_error is None:
                n_genes = sq_err.shape[1]
                total_squared_error = torch.zeros((n_genes, n_lineages), device=device)
                samples_per_lineage = torch.zeros(n_lineages, device=device)

            for l in range(n_lineages):
                l_mask = mask[:, l]
                
                if l_mask.sum() > 0:
                    # Sum the errors only for cells active on this lineage
                    total_squared_error[:, l] += sq_err[l_mask].sum(dim=0)
                    samples_per_lineage[l] += l_mask.sum()

    # Calculate mean per gene, per lineage
    mse_per_curve = total_squared_error / samples_per_lineage

    return mse_per_curve


def calculate_nll_per_gene(dataloader, model, loss_fn, device="cpu"):
    model.eval()
    total_nll = None

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            mu, theta, pi = model(X)
            
            loss = loss_fn(y, mu, theta, pi) 
            
            if total_nll is None:
                n_genes = loss.shape[1]
                total_nll = torch.zeros(n_genes, device=device)
                
            total_nll += loss.sum(dim=0)
            
    return total_nll


def calculate_aic(n_params, neg_log_likelihood):
    # AIC = 2k + 2 * NLL
    return 2 * n_params + 2 * neg_log_likelihood


def calculate_bic(n_params, neg_log_likelihood, n_samples):
    # BIC = k * ln(n) + 2 * NLL
    return n_params * np.log(n_samples) + 2 * neg_log_likelihood


def run_de(args, adata, pseudotime, weights):
    sim = args.sim
    data_dir = args.data_dir
    model_dir = args.model_dir
    fig_dir = args.fig_dir
    model_name = args.name
    dataset = args.dataset
    lineage = args.lineage

    model_path = os.path.join(model_dir, dataset, model_name)
    checkpoint = torch.load(model_path, weights_only=False)

    null_name = f"null_sim{sim}_all.pth"
    null_path = os.path.join(model_dir, dataset, null_name)
    null_checkpoint = torch.load(null_path, weights_only=False)
    
    model_type = checkpoint ["model"]
    input_dim = checkpoint["input_dim"]
    output_dim = checkpoint["output_dim"]
    model_gene = checkpoint["gene"]
    pt_min = checkpoint["pt_min"]
    pt_max = checkpoint["pt_max"]

    model = build_model(model_type, input_dim, output_dim)
    model.load_state_dict(checkpoint["state_dict"])

    model.eval()

    threshold = 0.8 
    pred_de = association_test(model, pseudotime, weights, model_gene, threshold, pt_min, pt_max)
    print(f"DE genes (association): {pred_de.shape[0]}")
    pred_de = information_criteria_test(checkpoint, null_checkpoint, threshold)
    print(f"DE genes (information): {pred_de.shape[0]}")