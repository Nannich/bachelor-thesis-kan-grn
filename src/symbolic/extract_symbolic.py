import sympy
import torch
import tempfile
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from kan.utils import ex_round, SYMBOLIC_LIB
from kan import KAN as PyKAN

from src.core.config import MODELS_DIR, RESULTS_DIR, DATA_RAW, ensure_dir
from src.symbolic.eval_symbolic import evaluate_equations
from src.symbolic.train_symbolic import train_deep_symbolic_kan

# Custom Sigmoid Function Registration for PyKAN Symbolic Library
def torch_sigmoid(x):
    return 1 / (1 + torch.exp(-x))

class sigmoid(sympy.Function):
    nargs = 1
    @classmethod
    def eval(cls, x):
        return None
    def _eval_derivative(self, symbol):
        return self.func(self.args[0]) * (1 - self.func(self.args[0]))
    def _eval_evalf(self, prec):
        arg_val = float(self.args[0].evalf(prec))
        import math
        return sympy.Float(1 / (1 + math.exp(-arg_val)), prec)

def sympy_sigmoid(x):
    return sigmoid(x)

# Register custom sigmoid function to PyKAN global math library
SYMBOLIC_LIB['sigmoid'] = (torch_sigmoid, sympy_sigmoid, 3, lambda x, y: (x, y))


def prune_small_coefficients(expr, thresh=0.05):
    """Zeros out any SymPy Float coefficient below a threshold."""
    return expr.subs({n: 0 for n in expr.atoms(sympy.Float) if abs(n) < thresh})


def extract_symbolic_grn(checkpoint, checkpoint_path, dataset_name, tmp_dir, custom_lib, output_fig_path=None, deep_config=None, skip_deep=False, prune=False):
    """Handles symbolic formula extraction from GRN models."""
    target_gene = checkpoint["target_gene"]
    predictor_names = checkpoint["predictor_names"]
    loss_mode = checkpoint["loss_mode"]
    X_numpy = checkpoint["X_numpy"]

    X_tensor = torch.tensor(X_numpy, dtype=torch.float32)
    in_dim = X_numpy.shape[1]

    if skip_deep:
        print(f"  Extracting directly from original KAN checkpoint weights.")
        width = checkpoint.get("width")
        if not width:
            out_dim = 3 if loss_mode == "zinb" else 1
            width = [in_dim] + checkpoint.get("hidden_layers", []) + [out_dim]
            
        kan_model = PyKAN(
            width=width, 
            grid=checkpoint.get("grid", 3), 
            k=checkpoint.get("k", 3), 
            device="cpu", 
            auto_save=False,
            ckpt_path=str(tmp_dir)
        )
        sd = {k.replace("kan.", ""): v for k, v in checkpoint["state_dict"].items()}
        kan_model.load_state_dict(sd)
    else:
        Y_numpy = checkpoint["Y_numpy"]
        Y_tensor = torch.tensor(Y_numpy, dtype=torch.float32)
        hidden_layers = deep_config if deep_config is not None else [2, 2]

        print(f"  Training symbolic KAN: {[in_dim] + hidden_layers + [3 if loss_mode == 'zinb' else 1]}")
        kan_model = train_deep_symbolic_kan(
            X_tensor, Y_tensor, hidden_layers=hidden_layers, ckpt_dir=tmp_dir,
            loss_mode=loss_mode, epochs=350, lr=0.01,
        )
    
    with torch.no_grad():
        kan_model(X_tensor)
        
    if loss_mode == "zinb":
        final_layer_idx = len(kan_model.width) - 1
        kan_model.remove_node(final_layer_idx, 1, mode='down')
        kan_model.remove_node(final_layer_idx, 2, mode='down')
    
    if prune:
        print("  Pruning model graph and low-weight edges...")
        kan_model = kan_model.prune()
        kan_model.prune_edge(threshold=0.05)    
    
    if output_fig_path:
        output_fig_path = Path(output_fig_path)
        ensure_dir(output_fig_path.parent)
        kan_model.plot(folder=str(output_fig_path.parent), beta=3, scale=2.0, in_vars=predictor_names, out_vars=[target_gene])
        plt.savefig(output_fig_path, bbox_inches="tight", dpi=300)
        plt.close()

    arch_dir = checkpoint_path.parent.name
    symbolic_model_dir = ensure_dir(MODELS_DIR / dataset_name / "symbolic" / arch_dir)
    
    base_prefix = "direct_" if skip_deep else "deep_"
    prefix = f"{base_prefix}pruned_" if prune else base_prefix
    deep_ckpt_path = symbolic_model_dir / f"{prefix}{checkpoint_path.name}"
    
    torch.save({
        "state_dict": {k: v.cpu() for k, v in kan_model.state_dict().items()},
        "target_gene": target_gene,
        "predictor_names": predictor_names,
        "loss_mode": loss_mode,
        "X_numpy": X_numpy,
        "width": kan_model.width,
        "grid": kan_model.grid,
        "k": kan_model.k,
        "is_symbolic_pruned": prune
    }, deep_ckpt_path)

    for param in kan_model.parameters():
        param.requires_grad = False
        
    kan_model.auto_symbolic(lib=custom_lib, weight_simple=0.5, r2_threshold=0.01)
    input_symbols = [sympy.Symbol(name) for name in predictor_names]
    formulas = kan_model.symbolic_formula(var=input_symbols)
    
    raw_formula = formulas[0][0]
    
    if prune:
        raw_formula = prune_small_coefficients(raw_formula, thresh=0.05)
    
    return target_gene, str(ex_round(raw_formula, 2))


def run_symbolic_pipeline(dataset_name, arch_name="log_log_l", skip_deep=False, prune=False):
    """Processes checkpoints, handles extraction variants, and saves expressions."""
    print(f"Symbolic Formula Extraction: {dataset_name} (grn | {arch_name})")
    
    target_model_dir = MODELS_DIR / dataset_name / "grn" / arch_name
    checkpoint_paths = sorted(list(target_model_dir.glob("*_checkpoint.pth")))
                       
    if not checkpoint_paths:
        print(f"No checkpoint found at: {target_model_dir}")
        return

    formulas_out_dir = ensure_dir(RESULTS_DIR / "symbolic" / dataset_name / "formulas")
    
    variant_token = "direct" if skip_deep else "deep"
    file_token = f"{arch_name}_{variant_token}_pruned" if prune else f"{arch_name}_{variant_token}"
    
    eq_csv_path = formulas_out_dir / f"{file_token}_equations.csv"
    output_fig_dir = ensure_dir(RESULTS_DIR / "figures" / dataset_name / "symbolic" / arch_name)
    
    extracted_records = []
    custom_lib = ['x', 'x^2', '1/x', '1/x^2', 'sqrt', 'exp', 'log', '0', 'sigmoid']

    for cp_path in checkpoint_paths:
        try:
            gene_symbol = cp_path.name.split("_checkpoint")[0]
            
            fig_suffix = "_pruned_symbolic_graph.png" if prune else "_symbolic_graph.png"
            fig_out_path = output_fig_dir / f"{gene_symbol}{fig_suffix}"
            
            print(f" Processing checkpoint weights for gene: {gene_symbol}...")
            checkpoint = torch.load(cp_path, map_location="cpu", weights_only=False)
            
            with tempfile.TemporaryDirectory() as tmp_dir:
                gene_name, formula = extract_symbolic_grn(
                    checkpoint=checkpoint, checkpoint_path=cp_path, dataset_name=dataset_name,
                    tmp_dir=tmp_dir, custom_lib=custom_lib, output_fig_path=fig_out_path, 
                    deep_config=[2, 2], skip_deep=skip_deep, prune=prune
                )
            
            extracted_records.append({"TargetGene": gene_name, "Equation": formula})
        except Exception as err:
            print(f"  Error {cp_path.name}: {err}")
            continue

    df_equations = pd.DataFrame(extracted_records)
    df_equations.to_csv(eq_csv_path, index=False)

    # Evaluate signs on ground-truth
    matched_dirs = list(DATA_RAW.rglob(f"**/{dataset_name}/ExpressionData.csv"))
    if not matched_dirs:
        return
        
    final_gt_path = None
    for parent in matched_dirs[0].parents:
        check_path = parent / "GroundTruthNetwork.csv"
        if check_path.exists():
            final_gt_path = check_path
            break

    if not final_gt_path:
        return

    eval_out_dir = ensure_dir(RESULTS_DIR / "symbolic" / dataset_name / "eval")
    evaluate_equations(eq_csv_path=eq_csv_path, ground_truth_path=final_gt_path, eval_out_dir=eval_out_dir, file_token=file_token)