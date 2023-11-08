from torch import torch
import numpy as np


def nontraining_mse(model, nontraining_data):

    if isinstance(nontraining_data, np.ndarray):
        nontraining_data = torch.tensor(nontraining_data)

    model.eval()
    with torch.no_grad():
        pred = model.predict(nontraining_data)

    cell_mse = ((np.array(nontraining_data) - pred.numpy()) ** 2).mean(axis=1)
    nt_mse = cell_mse.mean()

    return cell_mse, nt_mse


def log_mse(model, nontraining_data, logger, checkpoint_dir, step: str):

    cell_mse, nt_mse = nontraining_mse(model, nontraining_data)
    np.save(f"{checkpoint_dir}/{step}single_cell_mse.npy", cell_mse)

    logger.log_metrics({f"{step}/MSE": nt_mse})
