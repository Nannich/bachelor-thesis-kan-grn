import numpy as np
import pandas as pd
import torch
from pathlib import Path
from argparse import Namespace
from sklearn.metrics import roc_auc_score, average_precision_score

from src.core.config import DATA_RAW, MODELS_DIR, RESULTS_DIR, ensure_dir
from src.core.preprocessing import run_preprocessing
from src.grn.extract_grn import infer_grn_network
from src.trajectory.train_trajectory import run_trajectory

# Architectures to benchmark
ARCHITECTURES = [
    {"name": "log_log", "input_mode": "log", "target_mode": "log", "loss": "mse", "lag": 0.0, "traj_loss": "mse"},
    {"name": "smo_log", "input_mode": "smooth", "target_mode": "log", "loss": "mse", "lag": 0.0, "traj_loss": "mse"},
    {"name": "log_smo", "input_mode": "log", "target_mode": "smooth", "loss": "mse", "lag": 0.0, "traj_loss": "mse"},
    {"name": "smo_smo", "input_mode": "smooth", "target_mode": "smooth", "loss": "mse", "lag": 0.0, "traj_loss": "mse"},
    
    {"name": "log_log_l", "input_mode": "log", "target_mode": "log", "loss": "mse", "lag": 0.1, "traj_loss": "mse"},
    {"name": "smo_log_l", "input_mode": "smooth", "target_mode": "log", "loss": "mse", "lag": 0.1, "traj_loss": "mse"},
    {"name": "log_smo_l", "input_mode": "log", "target_mode": "smooth", "loss": "mse", "lag": 0.1, "traj_loss": "mse"},
    {"name": "smo_smo_l", "input_mode": "smooth", "target_mode": "smooth", "loss": "mse", "lag": 0.1, "traj_loss": "mse"},

    {"name": "smo_log_lz", "input_mode": "smooth", "target_mode": "log", "loss": "mse", "lag": 0.1, "traj_loss": "zinb"},
    {"name": "log_smo_lz", "input_mode": "log", "target_mode": "smooth", "loss": "mse", "lag": 0.1, "traj_loss": "zinb"},
    {"name": "smo_smo_lz", "input_mode": "smooth", "target_mode": "smooth", "loss": "mse", "lag": 0.1, "traj_loss": "zinb"},
]

def calculate_network_metrics(df_edges, gt_csv_path, gene_names):
    """
    Computes AUROC, AUPRC, and AUPRC ratio.
    """
    df_gt = pd.read_csv(gt_csv_path)
    df_gt.columns = [col.capitalize() for col in df_gt.columns]
    gt_edges = set(zip(df_gt["Gene1"].astype(str), df_gt["Gene2"].astype(str)))

    predicted_lookup = {}
    for _, row in df_edges.iterrows():
        edge_key = (str(row["Gene1"]), str(row["Gene2"]))
        predicted_lookup[edge_key] = abs(row["EdgeWeight"])

    y_true = []
    y_scores = []
    
    for g1 in gene_names:
        for g2 in gene_names:
            if g1 == g2:
                continue
            edge_key = (str(g1), str(g2))
            y_true.append(1 if edge_key in gt_edges else 0)
            y_scores.append(predicted_lookup.get(edge_key, 0.0))

    y_true = np.array(y_true)
    y_scores = np.array(y_scores)

    auroc = roc_auc_score(y_true, y_scores)
    auprc = average_precision_score(y_true, y_scores)
    prior_probability = np.mean(y_true)
    auprc_ratio = auprc / prior_probability if prior_probability > 0 else 1.0

    return auroc, auprc, auprc_ratio


def run_benchmark_grn(args):
    """
    Executes architectures, evaluates accuracy metrics, and exports summary tables.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root_search_path = Path(args.search_path)
    
    print(f"Scanning datasets at: {root_search_path}")
    discovered_datasets = []
    
    # Identify dataset directories containing required files
    for path in root_search_path.rglob("*"):
        if path.is_dir() and (path / "ExpressionData.csv").exists() and (path / "PseudoTime.csv").exists():
            discovered_datasets.append(path)
            
    if not discovered_datasets:
        print("No valid dataset directories located.")
        return

    print(f"Found {len(discovered_datasets)} datsets.")
    results_accumulator = []

    # Process all discovered dataset paths
    for dataset_path in sorted(discovered_datasets):
        dataset_name = dataset_path.name        
        dataset_type = dataset_path.parent.name 
        
        print(f"Dataset: {dataset_name} (Type: {dataset_type})")
        
        try:
            dataset_obj = run_preprocessing(dataset_name=dataset_name)
        except Exception as preprocess_err:
            print(f"Failed loading metrics for {dataset_name}, skipping: {preprocess_err}")
            continue

        gene_names = list(dataset_obj.gene_names)
        
        # Traverse up parent directories dynamically to find the GroundTruthNetwork.csv reference
        gt_path = None
        for parent in [dataset_path] + list(dataset_path.parents):
            check_path = parent / "GroundTruthNetwork.csv"
            if check_path.exists():
                gt_path = check_path
                break
            if parent == DATA_RAW:
                break
        
        trajectory_dir = MODELS_DIR / dataset_name / "trajectory"

        # Pre-train smooth trajectory curves once per individual dataset
        for loss_type in ["mse", "zinb"]:
            for gene_idx in range(len(gene_names)):
                gene_name = gene_names[gene_idx]
                trajectory_checkpoint_path = trajectory_dir / f"kan_{gene_name}_{loss_type}.pth"
                
                if trajectory_checkpoint_path.exists():
                    print(f"  Trajectory model for gene '{gene_name}' ({loss_type}) already exists. Skipping training.")
                    continue
                    
                traj_args = Namespace(
                    model="kan",
                    loss=loss_type,
                    gene=gene_idx,
                    hidden_layers=None,
                    epochs=500,
                    model_dir=trajectory_dir
                )
                
                run_trajectory(traj_args, dataset_obj)

        # Run each architecture
        for config in ARCHITECTURES:
            arch_name = config["name"]
            
            checkpoint_save_path = ensure_dir(MODELS_DIR / dataset_name / "grn" / arch_name)
            print(f" Running Architecture: {arch_name}")
            
            output_file_path = RESULTS_DIR / "grn" / dataset_name / arch_name / "rankedEdges.csv"
            
            try:
                # Skip recalculating existing files
                if output_file_path.exists():
                    print(f"  Found RankedEdges.csv, Skipping.")
                    df_edges = pd.read_csv(output_file_path, sep="\t")
                else:
                    df_edges = infer_grn_network(
                        dataset=dataset_obj,
                        input_mode=config["input_mode"],
                        target_mode=config["target_mode"],
                        loss_mode=config["loss"],
                        dt_val=config["lag"],
                        ground_truth_map=None,
                        trajectory_dir=trajectory_dir,
                        checkpoint_save_path=checkpoint_save_path,
                        device=device,
                        epochs=200,
                        lr=0.01,
                        lamb_l1=0.02,
                        traj_loss_mode=config["traj_loss"]
                    )
                    
                    if not df_edges.empty:
                        arch_out_dir = ensure_dir(RESULTS_DIR / "grn" / dataset_name / arch_name)
                        ranked_edges = df_edges.copy()
                        ranked_edges.to_csv(output_file_path, sep="\t", index=False)
                        print(f"  Saved edge list at: {output_file_path}")
                
                auroc, auprc, auprc_ratio = calculate_network_metrics(df_edges, gt_path, gene_names)
                
                results_accumulator.append({
                    "Dataset": dataset_name,
                    "Type": dataset_type,
                    "Architecture": arch_name,
                    "AUROC": auroc,
                    "AUPRC": auprc,
                    "AUPRC_Ratio": auprc_ratio
                })
                
            except Exception as loop_err:
                print(f" Error running {arch_name} on {dataset_name}: {loop_err}")
                continue

    df_all = pd.DataFrame(results_accumulator)
    
    # Resolve target output subdirectory from input path token
    folder_token = root_search_path.name
    benchmark_out_dir = ensure_dir(RESULTS_DIR / "benchmark" / folder_token)

    metrics_map = {
        "AUROC": "auroc",
        "AUPRC": "auprc",
        "AUPRC_Ratio": "auprc_ratio"
    }

    for label, column_key in metrics_map.items():
        # Save metrics for each individual dataset
        df_indiv_pivot = df_all.pivot(index="Architecture", columns="Dataset", values=label)
        df_indiv_pivot = df_indiv_pivot.reindex([a["name"] for a in ARCHITECTURES if a["name"] in df_indiv_pivot.index])
        
        indiv_file_path = benchmark_out_dir / f"{column_key}_individual.csv"
        df_indiv_pivot.to_csv(indiv_file_path)
        print(f"Saved Individual Dataset Spreadsheet at: {indiv_file_path}")

        # Save median metrics for each dataset group
        df_grouped_medians = df_all.groupby(["Architecture", "Type"])[label].median().reset_index()
        df_median_pivot = df_grouped_medians.pivot(index="Architecture", columns="Type", values=label)
        df_median_pivot = df_median_pivot.reindex([a["name"] for a in ARCHITECTURES if a["name"] in df_median_pivot.index])
        
        median_file_path = benchmark_out_dir / f"{column_key}_median.csv"
        df_median_pivot.to_csv(median_file_path)
        print(f"Saved Median Spreadsheet at: {median_file_path}")