import click
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_experiments.config import (
    load_config,
    save_config,
    check_experiment_root,
    check_experiment_name,
    check_trainer_params,
    check_mlflow_uri,
    check_checkpoint_params,
    keys,
    setup_mlflow_artifact_uri,
    load_env,
)
from pytorch_experiments.utils import flatten

from project_code.data.data import data_registry
from project_code.models import model_registry
from project_code.model_selection_analysis.nontraining_scores_pred import log_mse
from project_code.model_selection_analysis.get_latent_space import log_latent_space


def run(config):
    experiment_root, config = check_experiment_root(config)
    experiment_name, config = check_experiment_name(config)
    trainer_params, config = check_trainer_params(config)
    checkpoint_params, config = check_checkpoint_params(config)

    mlflow_uri, config = check_mlflow_uri(config)
    config = setup_mlflow_artifact_uri(config)
    load_env(config)

    save_config(experiment_root / "config.json", config)

    healthy_data_params = config["healthy_data"]
    nt_data_params = config["nt_data"]
    model_params = config[keys.MODEL_KEY]

    print(healthy_data_params)
    print(nt_data_params)

    model = model_registry.get(model_params)
    data_module = data_registry.get(healthy_data_params)

    checkpoint_callback = ModelCheckpoint(**checkpoint_params, dirpath=experiment_root)
    mlflogger = MLFlowLogger(experiment_name=experiment_name, tracking_uri=mlflow_uri)
    mlflogger.log_hyperparams(params=flatten(config))
    trainer = pl.Trainer(
        **trainer_params, logger=mlflogger, callbacks=[checkpoint_callback]
    )

    trainer.fit(model, datamodule=data_module)
    model.eval()
    mlflogger.experiment.log_artifact(
        mlflogger.run_id, checkpoint_callback.best_model_path
    )
    trainer.test(ckpt_path="best", datamodule=data_module)

    # Load best checkpoint model and datamodule
    ckpt = torch.load(checkpoint_callback.best_model_path)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    trained_scaler = ckpt["SimpleDataModule"]["trained_scaler"]
    nontraining_data_class = data_registry.get(nt_data_params)
    nontraining_data_class.set_scaler(trained_scaler)
    nontraining_data = nontraining_data_class.return_data_tensor()

    # Now evaluate what we want on the best model and with the trained scaler
    # Single cell MSE and blast/healthy prediction
    log_mse(
        model=model,
        nontraining_data=nontraining_data,
        logger=mlflogger,
        checkpoint_dir=checkpoint_callback.dirpath,
        step="NT",
    )

    # Encode cell and perform UMAP, save and log
    log_latent_space(
        model=model,
        data=nontraining_data,
        logger=mlflogger,
        checkpoint_dir=checkpoint_callback.dirpath,
        step="NT",
    )


@click.command()
@click.option("--config", "config_path")
def run_from_file(config_path):
    config = load_config(config_path)
    run(config)


if __name__ == "__main__":
    run_from_file()
