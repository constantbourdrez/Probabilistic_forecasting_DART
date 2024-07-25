import numpy as np
import pandas as pd
import torch as th

def calc_denominator(target, eval_points):
    return th.sum(th.abs(target * eval_points))

def quantile_loss(target, forecast, q: float, eval_points) -> float:
    return 2 * th.sum(
        th.abs((forecast - target) * eval_points * ((target <= forecast) * 1.0 - q))
    )

def calc_quantile_CRPS(target, forecast, eval_points):
    """
    target: (B, T, V), torch.Tensor
    forecast: (B, n_sample, T, V), torch.Tensor
    eval_points: (B, T, V): which values should be evaluated,
    """

    # target = target * scaler + mean_scaler
    # forecast = forecast * scaler + mean_scaler
    quantiles = np.arange(0.05, 1.0, 0.05)
    denom = calc_denominator(target, eval_points)
    CRPS = 0
    for i in range(len(quantiles)):
        q_pred = []
        for j in range(len(forecast)):
            q_pred.append(th.quantile(forecast[j : j + 1], quantiles[i], dim=1))
        q_pred = th.cat(q_pred, 0)
        q_loss = quantile_loss(target, q_pred, quantiles[i], eval_points)
        CRPS += q_loss / denom
    return CRPS.item() / len(quantiles)

def calc_mae(target, forecast, eval_points):
    """
    target: (B, T, V), torch.Tensor
    forecast: (B, n_sample, T, V), torch.Tensor
    eval_points: (B, T, V): which values should be evaluated,
    """
    return (th.sum(th.abs(target - forecast.mean(dim = 1)) * eval_points) / eval_points.sum()).item()
