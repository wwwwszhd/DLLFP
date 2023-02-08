import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MAE(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        loss = torch.mean(torch.abs(target - pred))
        return loss

class MSE(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        return torch.mean((target - pred)**2)

class MBE(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        loss = (target - pred).mean()
        return loss

class RAE(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        pred_mean = torch.mean(target)
        loss = torch.sum((target - pred).abs())/torch.sum((target - pred).abs())
        return loss

class RSE(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        target_mean = torch.mean(target)
        loss = torch.sum((target - pred)**2)/torch.sum((target - target_mean)**2)
        return loss

class MAPE(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        loss = torch.abs((target - pred) / target) * 100
        return loss

class RMSE(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forword(self, pred, target):
        mse = torch.mean((target - pred)**2)
        loss = torch.sqrt(mse)
        return loss

class MSLE(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forword(self, pred, target):
        epsilon = 1e-7
        log_pred = torch.log(torch.maximum(pred, epsilon) + 1.0)
        log_target = torch.log(torch.maximum(target, epsilon) + 1.0)
        square_error = (log_target - log_pred) ** 2
        loss = torch.mean(square_error)
        return loss

class RMSLE(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forword(self, pred, target):
        epsilon = 1e-7
        log_pred = torch.log(torch.maximum(pred, epsilon) + 1.0)
        log_target = torch.log(torch.maximum(target, epsilon) + 1.0)
        square_error = (log_target - log_pred) ** 2
        msle = torch.mean(square_error)
        loss = torch.sqrt(msle)
        return loss

class NRMSLE(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forword(self, pred, target):
        mse = torch.mean((target - pred) ** 2)
        rmse = torch.sqrt(mse)
        loss = rmse / torch.mean(target)
        return loss

class RRMSLE(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forword(self, pred, target):
        mse = torch.mean((target - pred) ** 2)
        square_target = torch.sum(pred ** 2)
        loss = torch.sqrt(mse / square_target)
        return loss

class HuberLoss(torch.nn.Module):
    def __init__(self, delta=1.0):
        super().__init__()
        self.delta = delta

    def forward(self, pred, target):
        error = target - pred
        abs_error = torch.abs(error)
        quadratic = torch.min(abs_error, torch.tensor(self.delta).to(abs_error.device))
        linear = (abs_error - quadratic)
        loss = 0.5 * quadratic ** 2 + self.delta * linear
        return loss.mean()

class Logcosh(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forword(self, pred, target):
        error = pred - target
        loss = torch.mean(torch.log(torch.cosh(error)))
        return loss

class QuantileLoss(torch.nn.Module):
    def __init__(self, quantile = 0.5):
        super().__init__()
        self.quantile = quantile
    def forward(self, pred, target):
        error = target - pred
        loss = torch.mean(torch.max(self.quantile * error, (self.quantile - 1) * error))
        return loss\

class CrossEntropy(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        loss = F.binary_cross_entropy_with_logits(pred, target)
        return loss

class FocalLoss(torch.nn.Module):
    def __init__(self, alpha = 2.0, gamma = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        BCE = F.binary_cross_entropy(pred, target)
        focal = self.alpha * (1 - pred) ** self.gamma * BCE
        return torch.mean(focal)

class KLDivergence(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        target = torch.clamp(target, min=1e-7, max=1.0)
        return F.kl_div(pred.log(), target)

class JSDivergence(torch.nn.Module):
    def __init__(self,):
        super().__init__()

    def forward(self, pred, target):
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        target = torch.clamp(target, min=1e-7, max=1.0)
        M = 0.5 * (pred + target)
        loss = 0.5 * F.kl_div(pred.log(), M) + 0.5 * F.kl_div(target.log(), M)
        return loss.mean()

