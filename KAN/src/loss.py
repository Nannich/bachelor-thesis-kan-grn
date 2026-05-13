import torch
from torch import nn
import torch.nn.functional as F

class ZINBLoss(nn.Module):
    def __init__(self, scale_factor=1.0, eps=1e-10, ridge_lambda=0.0, reduction='mean'):
        """
        Zero-Inflated Negative Binomial (ZINB) Loss
        Args:
            scale_factor (float): Scale factor applied to predictions.
            eps (float): Small value for numerical stability.
            ridge_lambda (float): Regularization weight for the zero-inflation probability (pi).
        """
        super(ZINBLoss, self).__init__()
        self.scale_factor = scale_factor
        self.eps = eps
        self.ridge_lambda = ridge_lambda
        self.reduction = reduction

    def forward(self, y_true, y_pred, theta, pi):
        """
        Compute the ZINB loss.
        Args:
            y_true (torch.Tensor): Ground truth counts (non-negative integers).
            y_pred (torch.Tensor): Predicted mean values (mu).
            theta (torch.Tensor): Dispersion parameter (shape parameter).
            pi (torch.Tensor): Zero-inflation probability (between 0 and 1).
        Returns:
            torch.Tensor: ZINB negative log-likelihood.
        """
        eps = self.eps
        y_true = y_true.float()

        # Clip y_pred, theta to avoid numerical issues
        y_pred = torch.clamp(y_pred, min=-10.0, max=12.0)
        theta = torch.clamp(theta, min=-10.0, max=12.0)

        y_pred = torch.exp(y_pred) * self.scale_factor  # Ensure mu > 0
        theta = torch.exp(theta)                        # Ensure theta > 0
        pi = torch.sigmoid(pi.float())                  # Ensure pi is in (0, 1)

        # Clip pi 
        pi = torch.clamp(pi, min=eps, max=1.0 - eps)

        # Negative binomial log-likelihood
        nb_case = (
            torch.lgamma(theta + eps)
            + torch.lgamma(y_true + 1.0)
            - torch.lgamma(y_true + theta + eps)
            + (theta + y_true) * torch.log(1.0 + (y_pred / (theta + eps)))
            + y_true * (torch.log(theta + eps) - torch.log(y_pred + eps))
        )

        # Zero-inflation log-likelihood for y_true = 0
        zero_nb = torch.pow(theta / (theta + y_pred + eps), theta)
        zero_case = -torch.log(pi + ((1.0 - pi) * zero_nb) + eps)

        # Combine cases: zero or NB
        result = torch.where(y_true < eps, zero_case, nb_case)

        # Add ridge penalty for pi
        ridge = self.ridge_lambda * torch.square(pi)
        result += ridge

        if self.reduction == 'mean':
            return torch.mean(result)
        elif self.reduction == 'sum':
            return torch.sum(result)
        elif self.reduction == 'none':
            return result

class MSEWrapperLoss(nn.Module):
    """
    A dummy wrapper that ignores theta and pi to train purely on MSE.
    Useful for baseline comparisons.
    """
    def __init__(self):
        super(MSEWrapperLoss, self).__init__()

    def forward(self, y_true, mu, theta, pi):
        # 1. Log transform the ground truth
        y_true_log1p = torch.log1p(y_true.float())
        
        # 2. Clamp and transform the prediction (just like in evaluation)
        mu_clamped = torch.clamp(mu, min=-10.0, max=12.0)
        y_pred_log1p = torch.log1p(torch.exp(mu_clamped))
        
        # 3. Calculate standard Mean Squared Error
        return F.mse_loss(y_pred_log1p, y_true_log1p)