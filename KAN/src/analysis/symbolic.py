import torch
import os
import sympy
from kan.utils import ex_round, SYMBOLIC_LIB
import numpy as np
import matplotlib.pyplot as plt

from src.model import build_model
from src.utils import *


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

SYMBOLIC_LIB['sigmoid'] = (torch_sigmoid, sympy_sigmoid, 2, lambda x, y: (x, y))

def symbolic_pykan(model, pseudotime, weights, fig_path, pt_min, pt_max):
    """
    Uses pykans in built functions to extract a symbolic formula from the KAN.
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
        'x', 'x^2', 'x^3', 'x^4', 'x^5', 
        '1/x', '1/x^2', '1/x^3', '1/x^4', '1/x^5', 
        'sqrt', 'x^0.5', 'x^1.5', '1/sqrt(x)', '1/x^0.5', 
        'exp', 'log', 'abs', '0', 'gaussian', 'sgn',
        'sigmoid',
        #'sin', 'cos', 
        'tan', 'tanh', 'arcsin', 'arccos', 'arctan', 'arctanh'
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


def symbolic_pysr(model, counts, pseudotime, weights, gene, model_gene, pt_min, pt_max, lineage):
    """
    Uses pysr to fit an function to the curve of the specific lineage.
    """

    from pysr import PySRRegressor

    is_single_gene = False if model_gene is None else True

    model.eval()
   
    predictions = predict_lineage_trajectories(pseudotime, weights, model, gene, pt_min, pt_max)
    pt_active_sorted, pt_input_scaled, y_pred = predictions[lineage]


    X_pysr = pt_input_scaled[:, lineage].reshape(-1, 1) # Format for PySR
    y_pred_flat = y_pred.flatten()

    pysr_model = PySRRegressor(
        niterations=100,
        binary_operators=["+", "*", "-", "/"],
        unary_operators=[
            "exp", 
            "inv(x) = 1/x",
            "sigmoid(x) = 1 / (1 + exp(-x))",
            #"gaussian(x) = exp(-x^2)",
            #"square(x) = x^2",
            #"log1p(x) = log(x + 1)", 
            #"relu(x) = max(0, x)"
        ],
        extra_sympy_mappings={
            "inv": lambda x: 1 / x,
            "sigmoid": lambda x: 1 / (1 + sympy.exp(-x)), 
            #"gaussian": lambda x: sympy.exp(-x**2),
            #"square": lambda x: x**2,
            #"log1p": lambda x: sympy.log(x + 1),
            #"relu": lambda x: sympy.Max(0, x)
        },
        variable_names=["x"],
        model_selection="best",
        random_state=0
    )

    pysr_model.fit(X_pysr, y_pred_flat)


def run_extraction(args, adata, pseudotime, weights):
    gene = args.gene
    data_dir = args.data_dir
    model_dir = args.model_dir
    fig_dir = args.fig_dir
    model_name = args.name
    dataset = args.dataset

    model_path = os.path.join(model_dir, model_name)
    
    checkpoint = torch.load(model_path, weights_only=False)
    model_type = checkpoint ["model"]
    input_dim = checkpoint["input_dim"]
    output_dim = checkpoint["output_dim"]
    model_gene = checkpoint["gene"]
    pt_min = checkpoint["pt_min"]
    pt_max = checkpoint["pt_max"]

    fig_path = os.path.join(fig_dir, "symbolic", f"{dataset}_gene{gene}.png")
    
    model = build_model(model_type, input_dim, output_dim)
    model.load_state_dict(checkpoint["state_dict"])

    model.eval()

    symbolic_pykan(model, pseudotime, weights, fig_path, pt_min, pt_max)
    #symbolic_pysr(model, counts, pseudotime, weights, gene, model_gene)
