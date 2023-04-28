from torch import torch
import numpy as np


def log_latent_space(model, data, logger, checkpoint_dir, step: str):

    print(step)

    if isinstance(data, np.ndarray):
        data = torch.tensor(data)

    with torch.no_grad():
        enc = model.endec.get_latent_space(data)

    np_enc = enc.numpy()

    # save encoding
    np.save(f"{checkpoint_dir}/{step}_encoded.npy", np_enc)

    # Log encoding
    logger.experiment.log_artifact(
        logger.run_id, f"{checkpoint_dir}/{step}_encoded.npy"
    )
