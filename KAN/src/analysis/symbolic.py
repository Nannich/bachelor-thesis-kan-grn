import torch
import torch.nn as nn
import torch.optim as optim
from kan import KAN as PyKAN
import os
import pandas as pd
import numpy as np
import sympy
from kan.utils import ex_round, SYMBOLIC_LIB
import numpy as np
import matplotlib.pyplot as plt
import pprint

from src.model import build_model
from src.utils import *
from src.analysis.grn import train_n_to_1_kan

def torch_sigmoid(x):
    return 1 / (1 + torch.exp(-x))

# Prevent sigmoid from being simplified/expanded
class sigmoid(sympy.Function):
    nargs = 1
    @classmethod
    def eval(cls, x):
        return None

def sympy_sigmoid(x):
    return sigmoid(x)

SYMBOLIC_LIB['sigmoid'] = (torch_sigmoid, sympy_sigmoid, 3, lambda x, y: (x, y))


def train_symbolic_kan(X_tensor, Y_tensor, hidden_layers=[1], epochs=400, lr=0.01, lamb_l1=0.00, grid=3, k=3):
    """
    Trains a KAN specifically for symbolic formula extraction.
    """
    if hidden_layers is None:
        hidden_layers = []
        
    in_dim = X_tensor.shape[1]
    
    width = [in_dim] + hidden_layers + [1]
    
    model = PyKAN(
        width=width, 
        grid=grid, 
        k=k,
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
        
        # L1 regularization strips away weak connections
        l1_loss = sum(torch.sum(torch.abs(param)) for param in model.parameters())
                
        loss = mse_loss + (lamb_l1 * l1_loss)
        loss.backward()
        optimizer.step()

    model.eval()
    return model

def extract_trajectory_symbolic(model, pseudotime, weights, fig_path, pt_min, pt_max):
    """
    Uses pykans built in functions to extract a symbolic formula from the KAN.
    """
    pt_scaled = (pseudotime - pt_min) / (pt_max - pt_min + 1e-8)

    trajectories = np.hstack((pt_scaled, weights))
    
    # Populate model
    model.eval()
    X_sample = torch.tensor(trajectories, dtype=torch.float32)
    
    with torch.no_grad():
        model(X_sample)

    # Pruning
    model.kan.remove_node(2, 1, mode='down')
    model.kan.remove_node(2, 2, mode='down')

    model.kan = model.kan.prune()
    model.kan.prune_edge(threshold=0.05)

    # Plotting the KAN
    n_lineages = weights.shape[1]
    input_names = [f"pt{i+1}" for i in range(n_lineages)] + [f"w{i+1}" for i in range(n_lineages)]
    
    output_names = ["$\\mu$", "$\\theta$", "$\\pi$"]

    model.kan.plot(
        folder='./figures', 
        beta=3, 
        scale=2.0, 
        varscale=0.33,
        in_vars=input_names, 
        out_vars=output_names,
    )
    
    plt.savefig(fig_path, bbox_inches="tight", dpi=300)
    plt.close()

    # Run symbolic regression to replace splines with math functions
    custom_lib = [
        'x', 'x^2', 
        '1/x', '1/x^2', 
        'sqrt', 
        'exp', 'log', '0', 
        'sigmoid'
    ]

    model.kan.auto_symbolic(lib=custom_lib, weight_simple=0.5, r2_threshold=0.01)

    # Generate the formula
    pt1, pt2, w1, w2 = sympy.symbols('pt1 pt2 w1 w2')
    input_symbols = [pt1, pt2, w1, w2] 
    formulas = model.kan.symbolic_formula(var=input_symbols)

    # Extract the formula for mu
    mu_formula = formulas[0][0] # type: ignore
    rounded_mu = ex_round(mu_formula, 1)
    
    print(rounded_mu)

def extract_symbolic_from_model(adata, model_dir, dataset, base_model_name, fig_dir, results_dir, target_gene=None):
    """ Extracts equations directly using data stored inside checkpoints of the grn models. """
    gene_names = list(adata.var_names)
    n_genes = len(gene_names)
    
    kan_dir = os.path.join(model_dir, "grn_models", dataset, base_model_name)
    if not os.path.exists(kan_dir):
        print(f"Error: No GRN models found at {kan_dir}. Run 'grn' first.")
        return

    out_fig_dir = os.path.join(fig_dir, "symbolic", "from_model", dataset, base_model_name)
    out_res_dir = os.path.join(results_dir, "symbolic", "from_model", dataset, base_model_name)
    os.makedirs(out_fig_dir, exist_ok=True)
    os.makedirs(out_res_dir, exist_ok=True)
    
    eq_file_path = os.path.join(out_res_dir, "symbolic_equations.txt")
    with open(eq_file_path, "w") as f:
        f.write(f"Symbolic Gene Interactions: {dataset} ({base_model_name})\n" + "="*60 + "\n\n")

    custom_lib = ['x', 'x^2', '1/x', '1/x^2', 'sqrt', 'exp', 'log', '0', 'sigmoid']
    genes_to_process = [target_gene] if target_gene and target_gene != "all" else gene_names

    for target_g in genes_to_process:
        model_path = os.path.join(kan_dir, f"{target_g}_kan.pth")
        if not os.path.exists(model_path): continue
            
        print(f"Extracting symbolic formula for {target_g}...")
        
        # Load checkpoint container
        checkpoint = torch.load(model_path, weights_only=False)
        grid_size = checkpoint.get("grid", 3)
        
        kan_model = PyKAN(width=[n_genes - 1, 1], grid=grid_size, k=3, device="cpu", auto_save=False)
        kan_model.load_state_dict(checkpoint["state_dict"])
        
        X_tensor = torch.tensor(checkpoint["X_numpy"], dtype=torch.float32)
        kan_model.eval()
        with torch.no_grad():
            kan_model(X_tensor)

        # Prune and map equations safely
        #kan_model = kan_model.prune()
        #kan_model.prune_edge(threshold=0.05)
        
        fig_path = os.path.join(out_fig_dir, f"{target_g}_interaction.png")
        input_gene_names = [name for name in gene_names if name != target_g]
        kan_model.plot(folder='./figures', beta=3, scale=2.0, in_vars=input_gene_names, out_vars=[target_g])
        plt.savefig(fig_path, bbox_inches="tight", dpi=300)
        plt.close()

        kan_model.auto_symbolic(lib=custom_lib, weight_simple=0.5, r2_threshold=0.01)
        input_symbols = [sympy.Symbol(name) for name in input_gene_names]
        formulas = kan_model.symbolic_formula(var=input_symbols)
        rounded_formula = ex_round(formulas[0][0], 2)
        
        with open(eq_file_path, "a") as f:
            f.write(f"{target_g} = {rounded_formula}\n\n")


def convert_grn_csv_to_edge_list(grn_csv_path: str, output_dir: str = "results/grn_list", threshold: float = 0.0) -> pd.DataFrame:
    """
    Converts the adjacemcy matrix to the BEELINE like list of edges.
    """
    df = pd.read_csv(grn_csv_path, index_col=0)
    edges = []
    
    for gene1 in df.index:
        for gene2 in df.columns:
            weight = df.loc[gene1, gene2]
            if abs(weight) > threshold:
                edge_type = "+" if weight > 0 else "-"
                edges.append({"Gene1": gene1, "Gene2": gene2, "Type": edge_type})
                
    os.makedirs(output_dir, exist_ok=True)
    out_name = os.path.basename(grn_csv_path).replace("_grn.csv", "_edges.csv")
    if not out_name.endswith("_edges.csv"):
        out_name = out_name.replace(".csv", "_edges.csv")
        
    out_path = os.path.join(output_dir, out_name)
    
    edge_df = pd.DataFrame(edges)
    edge_df.to_csv(out_path, index=False)
    print(f"Saved edge list to: {out_path}")
    
    return edge_df

def prep_tensors_for_target(adata, target_gene, predictor_genes):
    """Extracts expression matrices and converts them to PyTorch tensors."""

    
    gene_names = list(adata.var_names)
    expression_matrix = np.log1p(get_raw_counts(adata))
    
    target_idx = gene_names.index(target_gene)
    predictor_indices = [gene_names.index(p) for p in predictor_genes]
    
    Y_numpy = expression_matrix[:, [target_idx]]
    X_numpy = expression_matrix[:, predictor_indices]
    
    X_tensor = torch.tensor(X_numpy, dtype=torch.float32)
    Y_tensor = torch.tensor(Y_numpy, dtype=torch.float32)
    
    return X_tensor, Y_tensor


def extract_symbolic_from_edges(adata, edges_df, dataset, results_dir="./results/", fig_dir="./figures/"):
    """
    Trains a KAN model for each target gene using only its known predictors, then extracts and saves a symbolic formula.
    """
    out_res_dir = os.path.join(results_dir, "symbolic", "from_edges", dataset)
    out_fig_dir = os.path.join(fig_dir, "symbolic", "from_edges", dataset)
    
    os.makedirs(out_res_dir, exist_ok=True)
    os.makedirs(out_fig_dir, exist_ok=True)
    
    eq_file_path = os.path.join(out_res_dir, "symbolic_equations.txt")
    
    with open(eq_file_path, "w") as f:
        f.write(f"Symbolic Gene Interactions from Edges: {dataset}\n")

    custom_lib = ['x', '1/x', '1/x^2', 'sqrt', 'exp', 'log', '0', 'sigmoid']
    gene_names = list(adata.var_names)
    targets = edges_df['Gene2'].unique()
    
    for target_g in targets:
        # Extract unique predictor genes (Gene1) that point to this target
        predictors = list(set(edges_df[edges_df['Gene2'] == target_g]['Gene1'].tolist()))
        
        # Filter out the target gene itself from the predictors
        valid_predictors = [p for p in predictors if p in gene_names and p != target_g]
        
        if target_g not in gene_names or not valid_predictors:
            print(f"Skipping '{target_g}': No valid predictors (or only self-edges).")
            continue
            
        X_tensor, Y_tensor = prep_tensors_for_target(adata, target_g, valid_predictors)
        print(f"Training Symbolic KAN for target '{target_g}' with predictors: {valid_predictors}...")
        
        kan_model = train_symbolic_kan(
            X_tensor, 
            Y_tensor
        )
        
        with torch.no_grad():
            kan_model(X_tensor)
        
        #kan_model = kan_model.prune()
        
        fig_path = os.path.join(out_fig_dir, f"{target_g}_interaction.png")
        kan_model.plot(folder=out_fig_dir, beta=3, scale=2.0, in_vars=valid_predictors, out_vars=[target_g])
        plt.savefig(fig_path, bbox_inches="tight", dpi=300)
        plt.close()
        
        kan_model.auto_symbolic(lib=custom_lib, weight_simple=0.5, r2_threshold=0.01)
        
        input_symbols = [sympy.Symbol(name) for name in valid_predictors]
        formulas = kan_model.symbolic_formula(var=input_symbols)
        
        rounded_formula = ex_round(formulas[0][0], 2)
        
        with open(eq_file_path, "a") as f:
            f.write(f"{target_g} = {rounded_formula}\n\n")
            
    print(f"Equations saved to: {eq_file_path}")


USE_GROUND_TRUTH = True

def run_extraction(args, adata, pseudotime, weights):
    base_model_name = os.path.splitext(args.name)[0] if args.name else "raw"
    
    if USE_GROUND_TRUTH:
        # Use the raw ground truth network inside the dataset folder
        edge_file_path = os.path.join(args.data_dir, args.dataset, "GroundTruthNetwork.csv")
        
        if not os.path.exists(edge_file_path):
            print(f"Ground truth file not found at {edge_file_path}")
            return
            
        edges_df = pd.read_csv(edge_file_path)
        
    else:
        # Use downstream output from GRN pipeline
        dataset_group = os.path.basename(os.path.normpath(args.data_dir))
        
        grn_matrix_path = os.path.join("results", "grn", base_model_name, f"{dataset_group}_{args.dataset}_grn.csv")
        
        if os.path.exists(grn_matrix_path):
            print(f"Found inferred GRN matrix at: {grn_matrix_path}")
            edges_df = convert_grn_csv_to_edge_list(grn_matrix_path)
        elif args.name and os.path.exists(args.name):
            print(f"Using specific edge list file path: {args.name}")
            edges_df = pd.read_csv(args.name)
        else:
            extract_symbolic_from_model(
                adata=adata, 
                model_dir=args.model_dir, 
                dataset=args.dataset, 
                base_model_name=base_model_name, 
                fig_dir=args.fig_dir, 
                results_dir="./results/", 
                target_gene=args.gene
            )
            return

    rename_dict = {}
    for col in edges_df.columns:
        if col.lower() in ['gene1', 'gene2', 'type']:
            rename_dict[col] = col.capitalize()
        elif col.lower() == 'from': 
            rename_dict[col] = 'Gene1'
        elif col.lower() == 'to': 
            rename_dict[col] = 'Gene2'
            
    if rename_dict:
        edges_df = edges_df.rename(columns=rename_dict)

    extract_symbolic_from_edges(
        adata=adata,
        edges_df=edges_df,
        dataset=args.dataset,
        results_dir="./results/",
        fig_dir=args.fig_dir
    )