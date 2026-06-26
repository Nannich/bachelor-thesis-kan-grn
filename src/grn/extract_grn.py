import numpy as np
import pandas as pd
import torch
from pathlib import Path

from src.core.config import TABLES_DIR, DATA_RAW, ensure_dir, EXTERNAL_DIR, RESULTS_DIR
from src.grn.data_grn import get_lagged_expression, get_smoothed_expression
from src.grn.train_grn import train_n_to_1_kan

def extract_kan_signatures(model, X_tensor, loss_mode="mse"):
    """
    Extracts absolute edge attributions and direction signs from a shallow KAN.
    Uses direct spline attribution for the weight scores and gradients for 
    sign extraction.
    """
    model.eval()
    in_dim = X_tensor.shape[1]
    
    X_eval = X_tensor.clone().detach().requires_grad_(True)
    predictions = model(X_eval)
    target_vector = predictions[:, 0] if loss_mode == "zinb" else predictions
    
    grads = torch.autograd.grad(outputs=target_vector.sum(), inputs=X_eval)[0]
    
    # Mean of gradients determines directional sign
    mean_grads = grads.mean(dim=0).cpu().numpy()
    edge_signs = np.where(mean_grads >= 0, 1, -1)
    
    # Spline attribution score determines edge weight
    model.attribute(plot=False)
    edge_weights = model.edge_scores[0].detach().cpu().numpy().flatten()[:in_dim]
        
    return edge_weights, edge_signs


def infer_grn_network(dataset, input_mode, target_mode, loss_mode, 
                      dt_val, ground_truth_map, trajectory_dir, checkpoint_save_path, device,
                      epochs=200, lr=0.01, lamb_l1=0.02, traj_loss_mode="mse"):
    """
    Core loop for GRN inference. Handles data loading, centralized feature 
    masking via prior biological knowledge, and model checkpointing.
    """
    gene_names = list(dataset.gene_names)
    n_genes = len(gene_names)
    
    # Resolve matrix sources
    if input_mode == "smooth" or target_mode == "smooth":
        smooth_expression = get_smoothed_expression(dataset, trajectory_dir, traj_loss_mode)
    
    log_counts_in = smooth_expression if input_mode == "smooth" else np.log1p(dataset.raw_counts)
    log_counts_tgt = smooth_expression if target_mode == "smooth" else np.log1p(dataset.raw_counts)
    
    pseudotime = dataset.pseudotime
    lineage_assignment = dataset.lineage_assignment
    edges_accumulator = []

    # Train a single-output network configuration loop per gene
    for target_idx in range(n_genes):
        target_gene = gene_names[target_idx]
        print(f"[{target_idx + 1}/{n_genes}] Processing gene: {target_gene}...")

        # Generate full-width expression matrices
        X_full, Y_numpy = get_lagged_expression(
            log_counts_in, log_counts_tgt, pseudotime, lineage_assignment, target_idx, dt=dt_val
        )
        
        if X_full.shape[0] == 0:
            continue

        # If a ground truth map is provided filter the predictors of each gene
        if ground_truth_map and target_gene in ground_truth_map:
            predictor_names = [name for name in ground_truth_map[target_gene] if name in gene_names and name != target_gene]
        else:
            predictor_names = [name for name in gene_names if name != target_gene]

        predictor_indices = [gene_names.index(name) for name in predictor_names]
        X_numpy = X_full[:, predictor_indices]

        X_tensor = torch.tensor(X_numpy, dtype=torch.float32).to(device)
        Y_tensor = torch.tensor(Y_numpy, dtype=torch.float32).to(device)

        # Train the KAN
        kan_model = train_n_to_1_kan(
            X_tensor, Y_tensor, device=device,
            loss_mode=loss_mode, epochs=epochs, lr=lr, lamb_l1=lamb_l1
        )

        # Extract weight parameters and direction signs
        edge_weights, edge_signs = extract_kan_signatures(
            kan_model, X_tensor, loss_mode=loss_mode
        )

        # Store model checkpoints
        cpu_state_dict = {k: v.cpu() for k, v in kan_model.state_dict().items()}
        checkpoint = {
            "state_dict": cpu_state_dict,
            "target_gene": target_gene,
            "predictor_names": predictor_names,
            "grid": kan_model.grid,
            "k": kan_model.k,
            "hidden_layers": [],
            "loss_mode": loss_mode,
            "X_numpy": X_numpy,
            "Y_numpy": Y_numpy
        }
        torch.save(checkpoint, checkpoint_save_path / f"{target_gene}_checkpoint.pth")

        # Save edge list
        for i, source_gene in enumerate(predictor_names):
            final_weight = abs(edge_weights[i]) * edge_signs[i]
            edges_accumulator.append({
                "Gene1": source_gene,
                "Gene2": target_gene,
                "EdgeWeight": final_weight
            })
                

    # Sort edge list by descending absolute weight
    df_edges = pd.DataFrame(edges_accumulator)
    if not df_edges.empty:
        df_edges["AbsWeight"] = df_edges["EdgeWeight"].abs()
        df_edges = df_edges.sort_values(by="AbsWeight", ascending=False).drop(columns=["AbsWeight"]).reset_index(drop=True)

    return df_edges


def run_grn(args, dataset):
    """
    Unpacks args and handle saving the ranked edges spreadsheet.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dir = Path(args.model_dir)

    input_mode = getattr(args, "input_mode", "log")
    target_mode = getattr(args, "target_mode", "log")
    loss_mode = getattr(args, "loss", "mse")
    dt_val = getattr(args, "lag", 0.0)
    use_ground_truth = getattr(args, "ground_truth", False)

    in_token = "smo" if input_mode == "smooth" else "log"
    tgt_token = "smo" if target_mode == "smooth" else "log"
    
    suffix = "_l" if dt_val > 0 else ""
        
    gt_suffix = "_ground_truth" if use_ground_truth else ""
    experiment_name = f"{in_token}_{tgt_token}{suffix}{gt_suffix}"
    print(f"Executing GRN compilation pipeline track: {experiment_name}")

    # Load ground truth file if flag is set
    ground_truth_map = None
    if use_ground_truth:
        gt_path = list(DATA_RAW.rglob(f"**/{args.dataset}/GroundTruthNetwork.csv"))
        if gt_path:
            df_gt = pd.read_csv(gt_path[0])
            ground_truth_map = df_gt.groupby("Gene2")["Gene1"].apply(list).to_dict()

    trajectory_dir = model_dir.parent / "trajectory"
    checkpoint_save_path = ensure_dir(model_dir / experiment_name)

    df_edges = infer_grn_network(
        dataset=dataset, input_mode=input_mode, target_mode=target_mode,
        loss_mode=loss_mode, dt_val=dt_val,
        ground_truth_map=ground_truth_map, trajectory_dir=trajectory_dir,
        checkpoint_save_path=checkpoint_save_path, device=device,
        epochs=200, lr=0.01, lamb_l1=0.02
    )
    
    if df_edges.empty:
        print("Inference complete: No structural edges compiled to export.")
        return df_edges

    output_dir = ensure_dir(RESULTS_DIR / "grn" / args.dataset / experiment_name)
    output_file_path = output_dir / "rankedEdges.csv"
    
    ranked_edges = df_edges.copy()
    
    ranked_edges.to_csv(output_file_path, sep="\t", index=False)
    print(f"Saved Edge List: {output_file_path}")

    return df_edges