import numpy as np
import pandas as pd
import torch
from pathlib import Path
from argparse import Namespace

from src.core.config import MODELS_DIR, RESULTS_DIR, ensure_dir
from src.core.preprocessing import run_preprocessing
from src.trajectory.train_trajectory import run_trajectory
from src.trajectory.eval_trajectory import (
    calculate_mse_per_curve,
    calculate_nll_per_gene,
    calculate_aic,
    calculate_bic
)
from src.trajectory.zinb_models import build_kan_model, build_mlp_model
from src.trajectory.zinb_loss import ZINBLoss


CONFIGS = [
    {"model": "mlp", "loss": "mse", "hidden_layers": [10], "subdir": "mlp_small", "name": "MSE_MLP_SMALL"},
    {"model": "mlp", "loss": "mse", "hidden_layers": [16, 16], "subdir": "mlp_large", "name": "MSE_MLP_LARGE"},
    {"model": "kan", "loss": "mse", "hidden_layers": [1], "subdir": "", "name": "MSE_KAN"},
    {"model": "mlp", "loss": "zinb", "hidden_layers": [10], "subdir": "mlp_small", "name": "ZINB_MLP_SMALL"},
    {"model": "mlp", "loss": "zinb", "hidden_layers": [16, 16], "subdir": "mlp_large", "name": "ZINB_MLP_LARGE"},
    {"model": "kan", "loss": "zinb", "hidden_layers": [1], "subdir": "", "name": "ZINB_KAN"},
]

def run_benchmark_trajectory(args):
    """
    Executes trajectory fitting configurations across discovered datasets,
    evaluates performance metrics, and generates group-level and master summary tables.
    """
    root_search_path = Path(args.search_path)
    print(f"Scanning datasets for trajectory benchmarking at: {root_search_path}")
    
    discovered_datasets = []
    for path in root_search_path.rglob("*"):
        if path.is_dir() and (path / "ExpressionData.csv").exists() and (path / "PseudoTime.csv").exists():
            discovered_datasets.append(path)
            
    if not discovered_datasets:
        print("No valid dataset directories located.")
        return

    print(f"Found {len(discovered_datasets)} datasets.")
    
    loss_fn_zinb_eval = ZINBLoss(ridge_lambda=0.0, reduction='none')

    dataset_records = []

    for dataset_path in sorted(discovered_datasets):
        dataset_name = dataset_path.name
        dataset_type = dataset_path.parent.name 
        
        print(f"Dataset: {dataset_name} (Group: {dataset_type})")
        
        try:
            dataset_obj = run_preprocessing(dataset_name=dataset_name)
        except Exception as preprocess_err:
            print(f"Failed preprocessing {dataset_name}, skipping: {preprocess_err}")
            continue
            
        gene_names = list(dataset_obj.gene_names)
        n_samples = dataset_obj.pseudotime.shape[0]
        
        X = torch.tensor(dataset_obj.pseudotime, dtype=torch.float32)
        lineage_mask = torch.tensor(dataset_obj.lineage_assignment, dtype=torch.bool)
        n_lineages = lineage_mask.shape[1]
        
        trajectory_dir = MODELS_DIR / dataset_name / "trajectory"
        ensure_dir(trajectory_dir)
        
        for config in CONFIGS:
            model_type = config["model"]
            loss_type = config["loss"]
            arch_label = config["name"]
            hidden_layers = config["hidden_layers"]
            
            config_dir = trajectory_dir / config["subdir"] if config["subdir"] else trajectory_dir
            ensure_dir(config_dir)
            
            # Temporary pools to store single-gene results for a centralized dataset average
            gene_mse_pool = []
            gene_nll_pool = []
            gene_aic_pool = []
            gene_bic_pool = []
            measured_params = None
            
            for gene_idx in range(len(gene_names)):
                gene_name = gene_names[gene_idx]
                checkpoint_path = config_dir / f"{model_type}_{gene_name}_{loss_type}.pth"
                
                # Check for existing checkpoint
                if checkpoint_path.exists():
                    print(f"  Checkpoint found for {arch_label} (Gene: {gene_name}). Skipping training.")
                else:
                    traj_args = Namespace(
                        model=model_type,
                        loss=loss_type,
                        gene=gene_idx,
                        hidden_layers=hidden_layers,
                        epochs=500,
                        model_dir=config_dir
                    )
                    try:
                        run_trajectory(traj_args, dataset_obj)
                    except Exception as train_err:
                        print(f"  Error training {arch_label} for gene {gene_name}: {train_err}")
                        continue
                        
                # Extract optimization results and information criteria statistics
                try:
                    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                    actual_hidden = checkpoint.get("hidden_layers", [])
                    
                    if model_type == "kan":
                        model = build_kan_model(n_lineages, 1, actual_hidden)
                    elif model_type == "mlp":
                        model = build_mlp_model(n_lineages, 1, actual_hidden)
                    else:
                        continue
                        
                    model.load_state_dict(checkpoint["state_dict"])
                    model.eval()
                    
                    Y = torch.tensor(dataset_obj.raw_counts[:, [gene_idx]], dtype=torch.float32)
                    
                    # Calculate tracking error across all objective models
                    mse_tensor = calculate_mse_per_curve(model, X, Y, lineage_mask)
                    avg_mse = np.mean(mse_tensor.flatten().cpu().numpy())
                    
                    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                    measured_params = n_params  
                    
                    # Initialize probabilistic metrics as NaN for MSE setups
                    nll_val, aic_val, bic_val = np.nan, np.nan, np.nan
                    
                    if loss_type == "zinb":
                        total_nll = calculate_nll_per_gene(model, X, Y, loss_fn_zinb_eval)
                        nll_val = total_nll.item()
                        aic_val = calculate_aic(n_params, nll_val)
                        bic_val = calculate_bic(n_params, nll_val, n_samples)
                    
                    gene_mse_pool.append(avg_mse)
                    gene_nll_pool.append(nll_val)
                    gene_aic_pool.append(aic_val)
                    gene_bic_pool.append(bic_val)
                    
                except Exception as eval_err:
                    print(f"  Error evaluating checkpoint {checkpoint_path.name}: {eval_err}")
                    continue
            
            if gene_mse_pool and measured_params is not None:
                dataset_records.append({
                    "Dataset": dataset_name,
                    "Group": dataset_type,
                    "Architecture": arch_label,
                    "Parameters": measured_params,
                    "Avg_MSE": np.mean(gene_mse_pool),
                    "Negative_LogLikelihood": np.mean(gene_nll_pool) if loss_type == "zinb" else np.nan,
                    "AIC": np.mean(gene_aic_pool) if loss_type == "zinb" else np.nan,
                    "BIC": np.mean(gene_bic_pool) if loss_type == "zinb" else np.nan
                })

        
    df_all = pd.DataFrame(dataset_records)
    
    folder_token = root_search_path.name
    benchmark_out_dir = ensure_dir(RESULTS_DIR / "benchmark" / folder_token)
    
    metrics = ["Parameters", "Avg_MSE", "Negative_LogLikelihood", "AIC", "BIC"]
    arch_order = [c["name"] for c in CONFIGS]
    
    master_medians = {metric: {} for metric in metrics}
    
    # Generate Per-Metric Tables
    for metric in metrics:
        df_pivot = df_all.pivot_table(
            index="Architecture", 
            columns="Group", 
            values=metric, 
            aggfunc=np.median
        )
        
        # Enforce canonical ordering across rows
        df_pivot = df_pivot.reindex(arch_order)
        
        metric_file_path = benchmark_out_dir / f"trajectory_{metric.lower()}_group_median.csv"
        df_pivot.to_csv(metric_file_path)
        print(f"Saved Per-Metric Table ({metric}) at: {metric_file_path}")
        
        # Calculate intermediate group medians to populate the master sheet
        for arch in arch_order:
            if arch in df_pivot.index:
                row_values = df_pivot.loc[arch].dropna().values
                master_medians[metric][arch] = np.median(row_values) if len(row_values) > 0 else np.nan
            else:
                master_medians[metric][arch] = np.nan

    # Generate Master Summary Table
    master_records = []
    for arch in arch_order:
        record = {"Architecture": arch}
        for metric in metrics:
            record[metric] = master_medians[metric].get(arch, np.nan)
        master_records.append(record)
        
    df_master = pd.DataFrame(master_records).set_index("Architecture")
    master_file_path = benchmark_out_dir / "trajectory_master_table.csv"
    df_master.to_csv(master_file_path)
    print(f"Saved Master Table at: {master_file_path}")