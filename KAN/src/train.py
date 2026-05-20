import time
import torch
import os
import numpy as np
import copy

from src.dataloaders import get_dataloaders, get_eval_dataloader
from src.analysis.de import calculate_mse_per_curve, calculate_nll_per_gene, calculate_aic, calculate_bic
from src.model import build_model, MLP_HIDDEN_LAYERS, PYKAN_HIDDEN_LAYERS, EFFKAN_HIDDEN_LAYERS
from src.loss import ZINBLoss, MSEWrapperLoss

BATCH_SIZE = 256
EPOCHS = 2000
GRADIENT_CLIP_LIMIT = 5
PATIENCE = 12

TRAIN_CONFIG = {
    "mlp":    {"lr": 0.002, "wd": 6.5e-05},
    #"effkan": {"lr": 0.0021, "wd": 6.5e-05},
    "effkan": {"lr": 0.002, "wd": 6.5e-05},
    "pykan":  {"lr": 0.0024, "wd": 3.5e-6},
    "null":   {"lr": 0.001,  "wd": 0.0}
}



def train_loop(dataloader, model, loss_fn, optimizer, device, epoch, is_kan):
    model.train()
    total_loss = 0.0
    
    avg_mu, avg_theta, avg_pi = 0.0, 0.0, 0.0

    for batch_idx, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()

        # Update the KAN grid on the first batch of early epochs
        update_grid = is_kan and (epoch < 5 and batch_idx == 0)

        if is_kan:
            mu_logits, theta_logits, pi_logits = model(X, update_grid=update_grid)
        else:
            mu_logits, theta_logits, pi_logits = model(X)

        loss = loss_fn(y, mu_logits, theta_logits, pi_logits)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_LIMIT)
        optimizer.step()
        
        total_loss += loss.item()
        
        # Track raw network outputs
        with torch.no_grad():
            avg_mu += mu_logits.mean().item()
            avg_theta += theta_logits.mean().item()
            avg_pi += torch.sigmoid(pi_logits).mean().item()

    n_batches = len(dataloader)
    return total_loss / n_batches, avg_mu / n_batches, avg_theta / n_batches, avg_pi / n_batches


def test_loop(dataloader, model, loss_fn, device, is_kan):
    model.eval()
    total_test_loss = 0.0  

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            
            if is_kan:
                mu, theta, pi = model(X, update_grid=False)
            else:
                mu, theta, pi = model(X)
                
            loss = loss_fn(y, mu, theta, pi)
            total_test_loss += loss.item() 

    return total_test_loss / len(dataloader)


def run_training(args, adata, pseudotime, weights):
    model_type = args.model
    model_dir = args.model_dir
    target_gene = args.gene
    dataset = args.dataset

    config = TRAIN_CONFIG.get(model_type)
    lr = config["lr"]
    wd = config["wd"]

    train_dataloader, test_dataloader, input_dim, output_dim, pt_min, pt_max = get_dataloaders(
        adata, pseudotime, weights, target_gene, BATCH_SIZE
    )
    
     # Initialize the model
    model = build_model(model_type, input_dim, output_dim)
    checkpoint = {
        "input_dim": input_dim,
        "output_dim": output_dim,
        "model": model_type,
        "gene": target_gene,
        "state_dict": model.state_dict(),
        "pt_min": pt_min,
        "pt_max": pt_max,
        "hidden_layers": MLP_HIDDEN_LAYERS if model_type == "mlp" else (
                         EFFKAN_HIDDEN_LAYERS if model_type == "effkan" else PYKAN_HIDDEN_LAYERS),
        "wd": wd,
        "lr": lr,
        "mse": 0,
        "zinb_loss": float('inf')
    }

    gene_str = f"gene{target_gene}" if target_gene is not None else "all"
    # filename = f"MSE_{model_type}_{dataset}_{gene_str}.pth"
    filename = f"{model_type}_{dataset}_{gene_str}.pth"
    model_path = os.path.join(model_dir, filename)

    device = "cpu"
    model.to(device)
    print(f"Starting training on: {device}")
    training_start_time = time.time()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    #loss_fn = MSEWrapperLoss()
    loss_fn = ZINBLoss(ridge_lambda=0.11)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    is_kan = model_type in ["effkan", "pykan"]

    # Early Stopping Setup
    for t in range(EPOCHS):
        train_loss, a_mu, a_th, a_pi = train_loop(train_dataloader, model, loss_fn, optimizer, device, t, is_kan)
        val_loss = test_loop(test_dataloader, model, loss_fn, device, is_kan)

        if t % 5 == 0:
            print(f"Epoch [{t+1}/{EPOCHS}] | Train Loss: {train_loss:.4f} | Test: {val_loss:.4f} | "
                  f"Raw μ: {a_mu:.2f}, Raw θ: {a_th:.2f}, π: {a_pi:.2f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            checkpoint["state_dict"] = copy.deepcopy(model.state_dict())
            checkpoint["zinb_loss"] = best_val_loss
            torch.save(checkpoint, model_path)
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= PATIENCE:
            print(f"Early stopping triggered after {t+1} epochs.")
            checkpoint["epochs"] = t + 1
            torch.save(checkpoint, model_path)
            break

    training_end_time = time.time()
    total_duration = training_end_time - training_start_time
    
    print("-" * 30)
    print(f"TOTAL TRAINING TIME: {total_duration:.2f} seconds")
    print(f"AVERAGE TIME PER EPOCH: {total_duration / (t+1):.2f} seconds")
    print("-" * 30)

    # Add Evaluation metrics to checkpoint
    checkpoint = torch.load(model_path, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])    
    
    full_dataloader = get_eval_dataloader(
        adata, pseudotime, weights, pt_min, pt_max, target_gene, batch_size=256
    )
    
    mse_per_curve = calculate_mse_per_curve(full_dataloader, model, device)
    
    unreduced_loss_fn = ZINBLoss(reduction='none') 
    total_nll_tensor = calculate_nll_per_gene(full_dataloader, model, unreduced_loss_fn, device)
    
    # Number of parameters: k
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_genes = total_nll_tensor.shape[0]

    # Distribute k across the number of genes
    k_per_gene = n_params / n_genes
    n_samples = len(full_dataloader.dataset)

    # Calculate global bic and aic
    global_nll = total_nll_tensor.sum().item()
    global_k = n_params
    checkpoint["global_aic"] = 2 * global_k + 2 * global_nll
    checkpoint["global_bic"] = global_k * np.log(n_samples) + 2 * global_nll

    checkpoint["mse"] = mse_per_curve.cpu()
    checkpoint["aic"] = calculate_aic(k_per_gene, total_nll_tensor).cpu()
    checkpoint["bic"] = calculate_bic(k_per_gene, total_nll_tensor, n_samples).cpu()
    
    torch.save(checkpoint, model_path)