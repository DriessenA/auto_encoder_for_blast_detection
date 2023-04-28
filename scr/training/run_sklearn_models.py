import click
import mlflow
from pytorch_experiments.config import (
    load_config,
    save_config,
    check_experiment_root,
    check_experiment_name,
    check_mlflow_uri,
    check_checkpoint_params,
    keys,
    setup_mlflow_artifact_uri,
    load_env,
)
from pytorch_experiments.utils import flatten
from project_code.data.data import data_registry
from project_code.data.utils import asinh_transform, get_co_factors
from project_code.models import model_registry
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import MinMaxScaler
import pickle


def run(config):
    experiment_root, config = check_experiment_root(config)
    experiment_name, config = check_experiment_name(config)
    checkpoint_params, config = check_checkpoint_params(config)

    expression = config["expr_or_error"]

    mlflow_uri, config = check_mlflow_uri(config)
    config = setup_mlflow_artifact_uri(config)
    load_env(config)

    save_config(experiment_root / "config.json", config)

    dataset_params = config["sklearn_data"]
    model_params = config[keys.MODEL_KEY]

    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(experiment_name)
    mlflow.log_params(params=flatten(config))
    mlflow.sklearn.autolog()

    model = model_registry.get(model_params)
    X_train, y_train, X_val, y_val, X_test, y_test = data_registry.get(dataset_params)

    if expression:
        # Transform with cofactors
        X_train_transformed = asinh_transform(X_train, cf_dict=get_co_factors())
        X_val_transformed = asinh_transform(X_val, cf_dict=get_co_factors())

        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train_transformed)
        X_val = scaler.transform(X_val_transformed)

        pickle.dump(scaler, open(f"{experiment_root}/scaler.pkl", "wb"))
        mlflow.log_artifact(f"{experiment_root}/scaler.pkl")

    model.fit(X_train, y_train)
    val_pred = model.predict(X_val)
    mlflow.sklearn.eval_and_log_metrics(
        model, X_val, y_val, prefix="val_", sample_weight=None, pos_label=1
    )
    if X_test is not None:
        if expression:
            X_test_transformed = asinh_transform(X_test, cf_dict=get_co_factors())
            X_test = scaler.transform(X_test_transformed)
        mlflow.sklearn.eval_and_log_metrics(
            model,
            X_test,
            y_test,
            prefix="test_",
            sample_weight=None,
            pos_label=1,
        )
    accuracy_score(val_pred, y_val)
    f1_score(val_pred, y_val)


@click.command()
@click.option("--config", "config_path")
def run_from_file(config_path):
    config = load_config(config_path)
    run(config)


if __name__ == "__main__":
    run_from_file()
