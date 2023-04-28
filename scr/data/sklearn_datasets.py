import pandas as pd
import numpy as np
from typing import Union, List


def setup_sklearn_datasets(
    train_file: str,
    val_file: str,
    test_file: Union[str, None],
    input_cols: List,
    target: Union[np.ndarray, str, None] = None,
    batch_size=512,
    num_workers=1,
    random_state=11,
):
    train = pd.read_csv(train_file, index_col=0)
    val = pd.read_csv(val_file, index_col=0)

    if test_file is not None:
        test = pd.read_csv(test_file, index_col=0)

    if target is None:
        y_train = [1] * train.shape[0]
        y_val = [1] * val.shape[0]

        X_train = train[input_cols]
        X_val = val[input_cols]

        if test_file is not None:
            y_test = [1] * test.shape[0]
            X_test = test[input_cols]

    # So here get the input and the target
    # If the target name occurs in the column names
    elif target in train.columns:
        X_train = train[input_cols]
        y_train = train.pop(target)
        X_val = val[input_cols]
        y_val = val.pop(target)

        if test_file is not None:
            X_test = test[input_cols]
            y_test = test.pop(target)
    else:
        X_train = train[input_cols]
        y_train = target[0]
        X_val = val[input_cols]
        y_val = target[1]

        if test_file is not None:
            X_test = test[input_cols]
            y_test = target[2]

    if test_file is None:
        X_test = None
        y_test = None

    return X_train, y_train, X_val, y_val, X_test, y_test
