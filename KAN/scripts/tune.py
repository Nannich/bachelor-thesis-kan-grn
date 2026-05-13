import argparse
import optuna
import optuna.visualization as vis
from torch import optim
import os
import copy
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.preprocessing import run_preprocessing
from src.dataloaders import get_dataloaders, get_eval_dataloader
from src.utils import *
from src.loss import ZINBLoss
from src.train import train_loop, test_loop
from src.analysis.de import calculate_mse_per_curve
import src.model as my_models

optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective(trial, args, adata, pseudotime, weights):
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    wd = trial.suggest_float("wd", 1e-6, 1e-1, log=True)
    
    if args.model == "mlp":
        n_layers = trial.suggest_int("n_layers", 2, 6)        
        width = trial.suggest_int("width", 64, 512, log=True) 
        my_models.MLP_HIDDEN_LAYERS = [width] * n_layers
        is_kan = False
    else:
        n_hidden_layers = trial.suggest_int("kan_depth", 1, 3) 
        hidden_width = trial.suggest_int("kan_width", 2, 32, log=True)
        my_models.EFFKAN_HIDDEN_LAYERS = [hidden_width] * n_hidden_layers
        
        grid = trial.suggest_int("grid_size", 3, 10)
        my_models.EFFKAN_GRID_SIZE = grid
        is_kan = True

    train_dl, test_dl, in_dim, out_dim, pt_min, pt_max = get_dataloaders(
        adata, pseudotime, weights, target_gene=None, batch_size=256
    )
    
    device = "cpu"
    model = my_models.build_model(args.model, in_dim, out_dim).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = ZINBLoss(ridge_lambda=0.11)
    
    epochs = 250
    best_val_loss = float('inf')
    

    for epoch in range(epochs):
        train_loop(train_dl, model, loss_fn, optimizer, device, epoch, is_kan)
        val_loss = test_loop(test_dl, model, loss_fn, device, is_kan)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            
        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
            
    # Evaluate the model on the curve MSE instead of the ZINB loss
    model.load_state_dict(best_state)
    
    full_dataloader = get_eval_dataloader(
        adata, pseudotime, weights, pt_min, pt_max, target_gene=None, batch_size=256
    )
    
    mse_tensor = calculate_mse_per_curve(full_dataloader, model, device)

    mean_mse = mse_tensor.mean().item()
    
    return mean_mse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data/")
    parser.add_argument("--dataset", type=str, default="paul")
    parser.add_argument("--model", type=str, choices=["effkan", "mlp"], required=True)
    parser.add_argument("--timeout", type=int, default=5400, help="Time limit in seconds")
    args = parser.parse_args()

    adata, pseudotime, weights = run_preprocessing(args)
    
    study = optuna.create_study(
        study_name=f"{args.model}_mse_optimization",
        direction="minimize",
        storage="sqlite:///results/optuna_mse_study.db",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner() 
    )
    
    study.optimize(
        lambda trial: objective(trial, args, adata, pseudotime, weights), 
        timeout=args.timeout
    )

    print(f"Best MSE: {study.best_value:.4f}")
    for key, value in study.best_trial.params.items():
        print(f"  {key}: {value}")

    plot_dir = f"./plots/optuna/{args.model}"
    os.makedirs(plot_dir, exist_ok=True)

    vis.plot_optimization_history(study).write_html(f"{plot_dir}/history.html")
    vis.plot_parallel_coordinate(study).write_html(f"{plot_dir}/parallel_coords.html")
    vis.plot_param_importances(study).write_html(f"{plot_dir}/importance.html")


if __name__ == "__main__":
    main()



"""
MLP

ZINB loss

Best Validation Loss: 0.4787
  lr: 0.0004043342026046728
  wd: 7.69767453786849e-05
  n_layers: 4
  width: 85

TOTAL TRAINING TIME: 237.29 seconds
AVERAGE TIME PER EPOCH: 0.89 seconds

MSE

Best Mean Squared Error: 0.4055
  lr: 0.002134607088212705
  wd: 6.467956783803012e-05
  n_layers: 3
  width: 489

TOTAL TRAINING TIME: 109.01 seconds
AVERAGE TIME PER EPOCH: 1.33 seconds


KAN

ZINB loss

Best Validation Loss: 0.4617
  lr: 0.002393540794630194
  wd: 3.4924953150543754e-06
  kan_depth: 1
  kan_width: 7

TOTAL TRAINING TIME: 164.87 seconds
AVERAGE TIME PER EPOCH: 1.02 seconds


MSE

Best Mean Squared Error: 0.3123
  lr: 0.0001585765608357316
  wd: 0.006687694813375173
  kan_depth: 3
  kan_width: 24
  grid_size: 7

TOTAL TRAINING TIME: 589.43 seconds
AVERAGE TIME PER EPOCH: 1.27 seconds
"""