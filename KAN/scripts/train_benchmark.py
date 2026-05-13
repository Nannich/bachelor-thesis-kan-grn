import os
import time
import torch
import pandas as pd
import numpy as np
from argparse import Namespace
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.preprocessing import run_preprocessing
from src.train import run_training
from src.model import build_model

def main():
    runs_per_model = 10
    models_to_test = ["mlp", "effkan"]
    dataset_name = "paul"
    target_genes = [1040, 1670, 619] # Elane, Klf1, Cd34
    
    benchmark_model_dir = "./checkpoints/benchmark/"
    os.makedirs(benchmark_model_dir, exist_ok=True)
    
    # Mock args for the existing functions
    args = Namespace(
        dataset=dataset_name,
        data_dir="./data/",
        model_dir=benchmark_model_dir,
        gene=None, 
    )
    
    adata, pseudotime, weights = run_preprocessing(args)
    
    results = []
    
    # Benchmark Loop
    for model_type in models_to_test:
        args.model = model_type
        
        for run in range(1, runs_per_model + 1):
            print(f"  Run {run}/{runs_per_model} | Model: {model_type}")
            
            # Start timer and run training
            start_time = time.time()
            run_training(args, adata, pseudotime, weights)
            end_time = time.time()
            total_time = end_time - start_time
            
            # Load the checkpoint 
            model_path = os.path.join(args.model_dir, f"{model_type}_{dataset_name}_all.pth")
            checkpoint = torch.load(model_path, weights_only=False)
            
            # Extract Tensors
            mse_tensor = checkpoint.get("mse", torch.zeros((adata.n_vars, 2)))
            aic_tensor = checkpoint.get("aic", torch.zeros(adata.n_vars))
            bic_tensor = checkpoint.get("bic", torch.zeros(adata.n_vars))
            
            # Calculate total parameters by rebuilding the model
            model = build_model(checkpoint["model"], checkpoint["input_dim"], checkpoint["output_dim"])
            total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            # Extract Epochs and calculate time per epoch
            epochs_converged = checkpoint.get("epochs", np.nan)
            time_per_epoch = total_time / epochs_converged if not np.isnan(epochs_converged) else np.nan
            
            # Build the row dictionary
            run_data = {
                "Model": model_type.upper(),
                "Run": run,
                "Parameters": total_params, 
                "Epochs_Converged": epochs_converged,
                "Time_Per_Epoch": time_per_epoch,
                "Time_Seconds": total_time,
                "ZINB_Loss": checkpoint.get("zinb_loss", np.nan),
                "Global_MSE": mse_tensor.mean().item(),
                "Global_AIC": checkpoint.get("global_aic", np.nan),
                "Global_BIC": checkpoint.get("global_bic", np.nan),
            }
            
            # Extract target gene specific metrics
            for g in target_genes:
                run_data[f"Gene_{g}_AIC"] = aic_tensor[g].item()
                run_data[f"Gene_{g}_BIC"] = bic_tensor[g].item()
                run_data[f"Gene_{g}_MSE_L1"] = mse_tensor[g, 0].item()
                run_data[f"Gene_{g}_MSE_L2"] = mse_tensor[g, 1].item()
                
            results.append(run_data)
            
    # Save and Aggregate Results
    df = pd.DataFrame(results)
    csv_path = "results/train_benchmark_results_raw.csv"
    df.to_csv(csv_path, index=False)
    print(f"Raw benchmark data saved to: {csv_path}")
    
    # Calculate Mean and Standard Deviation
    df_numeric = df.drop(columns=["Run"])
    df_summary = df_numeric.groupby("Model").agg(['mean', 'std'])
    
    # Flatten columns for the CSV
    df_csv = df_summary.copy()
    df_csv.columns = [f"{col[0]}_{col[1]}" for col in df_csv.columns]
    df_csv = df_csv.reset_index()
    
    avg_csv_path = "results/train_benchmark_results_avg.csv"
    df_csv.to_csv(avg_csv_path, index=False)
    

if __name__ == "__main__":
    main()