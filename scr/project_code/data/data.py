from typing import List, Union
from pytorch_experiments.data.factory import DatamoduleRegistry
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
from torch import torch
import pandas as pd
import numpy as np
from .utils import (
    asinh_transform,
    SimpleDataModule,
    get_co_factors,
)
from .sklearn_datasets import setup_sklearn_datasets


class CytofDataset(Dataset):
    def __init__(self, X: pd.DataFrame, y, scaler: None):
        """
        Parameters
        ----------
        df:
            Dataframe of expression values, cells as rows and markers as columns
        train_cols:
           List of column names that are used as features for training
        target:
            List of either column names to use as target,
            or list with same length as df with target values
        cofactor_transform:
           Indicate whether or not to perform asinh
           transformation with hardcoded cofactors
        """
        self.X = X
        self.y = y

        if scaler is not None:
            self.X = scaler.transform(self.X)

        self.X = np.array(self.X, dtype=np.float32)
        self.y = np.array(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]

    def __len__(self):
        return self.X.shape[0]


def setup_cytof_datamodule(
    train_file: str,
    val_file: str,
    test_file: str,
    input_cols: List,
    target: Union[np.ndarray, str, None] = None,
    scaler=MinMaxScaler,
    batch_size=512,
    num_workers=1,
    cofactor_transform=True,
    random_state=11,
):
    train = pd.read_csv(train_file, index_col=0)
    val = pd.read_csv(val_file, index_col=0)
    test = pd.read_csv(test_file, index_col=0)

    if target is None:
        y_train = [1] * train.shape[0]
        y_val = [1] * val.shape[0]
        y_test = [1] * test.shape[0]

        X_train = train[input_cols]
        X_val = val[input_cols]
        X_test = test[input_cols]

    # So here get the input and the target
    # If the target name occurs in the column names
    elif target in train.columns:
        X_train = train[input_cols]
        y_train = train.pop[target]
        X_val = val[input_cols]
        y_val = val.pop[target]
        X_test = test[input_cols]
        y_test = test.pop[target]
    else:
        X_train = train[input_cols]
        y_train = target[0]
        X_val = val[input_cols]
        y_val = target[1]
        X_test = test[input_cols]
        y_test = target[2]

    # Do transformation of the input
    if cofactor_transform:
        X_train = asinh_transform(X_train, cf_dict=get_co_factors())
        X_val = asinh_transform(X_val, cf_dict=get_co_factors())
        X_test = asinh_transform(X_test, cf_dict=get_co_factors())

    # Train scaler on training data
    trained_scaler = scaler().fit(X_train)

    # Pass to each get_cytof_dataset function
    ds_train = CytofDataset(X=X_train, y=y_train, scaler=trained_scaler)
    ds_val = CytofDataset(X=X_val, y=y_val, scaler=trained_scaler)
    ds_test = CytofDataset(X=X_test, y=y_test, scaler=trained_scaler)

    return SimpleDataModule(
        train=ds_train,
        val=ds_val,
        test=ds_test,
        trained_scaler=trained_scaler,
        batch_size=batch_size,
        num_workers=num_workers,
    )


class NonTrainingData:
    def __init__(self, file_path: str, cofactor_transform: bool = True):
        self.file_path = file_path
        self.cofactor_transform = cofactor_transform

    def set_scaler(self, scaler):
        self.scaler = scaler

    def return_data_tensor(self):
        data = pd.read_csv(self.file_path, index_col=0)

        if self.cofactor_transform:
            transformed_data = asinh_transform(data, cf_dict=get_co_factors())
        scaled_data = self.scaler.transform(transformed_data)

        data_tensor = torch.tensor(scaled_data)

        return data_tensor


def setup_non_training_data(file_path, cofactor_transform):
    nt_class = NonTrainingData(file_path, cofactor_transform=cofactor_transform)

    return nt_class


data_registry = DatamoduleRegistry()
data_registry.register("cytof", setup_cytof_datamodule)
data_registry.register("non_training", setup_non_training_data)
data_registry.register("sklearn", setup_sklearn_datasets)
