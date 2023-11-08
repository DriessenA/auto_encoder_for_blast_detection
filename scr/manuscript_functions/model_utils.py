# import MLFlow
import re
import json
from sklearn.preprocessing import MinMaxScaler
from project_code.models import model_registry
from project_code.data.utils import get_co_factors, asinh_transform
import torch
import pandas as pd
import numpy as np
import umap.umap_ as umap
from sklearn.metrics import (
    roc_curve,
)
from typing import List


# MLFlow  ######################################
def get_mlflow_ckpt_path(run_id, client):
    for art in client.list_artifacts(run_id=run_id):
        matches = re.findall("'.+.ckpt'", str(art))

        if len(matches) > 1:
            dir_ = []
            for ckpt_dir in matches:
                m = re.sub("'", "", ckpt_dir)
                dir_.append(m)
                print(m)

        elif len(matches) == 1:
            dir_ = re.sub("'", "", matches[0])
            print(dir_)

    return dir_


def find_and_load_checkpoint(
    run_id, run_dir, client, AE, old_CV, ckpt_name: str = None
):
    config_file = f"{run_dir}/config.json"

    if ckpt_name is None:
        ckpt_name = get_mlflow_ckpt_path(run_id, client)
    checkpoint = f"{run_dir}/{ckpt_name}"

    with open(config_file, "r") as configfile:
        config = json.load(configfile)

    if AE:
        config["models"]["submodels"]["encoder_decoder"]["model"] = "FlexibleAE"

    model = model_registry.get(config["models"])
    ckpt = torch.load(checkpoint, map_location=torch.device("cpu"))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    if not old_CV:
        scaler = ckpt["SimpleDataModule"]["trained_scaler"]
    else:
        train_file = config["healthy_data"]["dataset_params"]["train_file"]
        trans_train = asinh_transform(
            pd.read_csv(
                train_file,
                index_col=0,
            ),
            get_co_factors(),
        )
        scaler = MinMaxScaler()
        scaler.fit(trans_train)

    return model, scaler


# Data utils
def preprocess(data, scaler, do_transform):

    if do_transform:
        transformed_data = asinh_transform(data, get_co_factors())
    else:
        transformed_data = data.copy()
    scaled_data = scaler.transform(transformed_data)

    return scaled_data


def get_run_meta(
    runs,
    run_id,
    client,
    expr_file,
    meta_file,
    thresh_mix_file,
    thresh_mix_meta,
    inc_rel_error: bool = False,
    old_CV: bool = False,
    transform_syn_mix: bool = False,
    run_dir: str = None,
):
    if isinstance(meta_file, str):
        meta = pd.read_csv(meta_file, index_col=0)
    elif isinstance(meta_file, pd.DataFrame):
        meta = meta_file.copy()
    else:
        raise Exception(
            "meta_file must either be a str with meta_file path or a pandas dataframe"
        )
    if run_dir is None:
        run_dir = runs.loc[runs["Run ID"] == run_id, "experiment_root"].values[0]
    AE = False

    if old_CV:
        if (
            runs.loc[runs["Run ID"] == run_id, "models.submodels.encoder_decoder.model"]
            == "FlexibleAE"
        ).values[0]:
            print("Using Flexible AE")
            AE = True
        else:
            AE = False

    try:
        sc_mse = np.load(f"{run_dir}/NT_single_cell_mse.npy")
        sc_mse_pred = np.load(f"{run_dir}/NT_single_cell_hb_pred.npy")
    except FileNotFoundError:
        sc_mse, sc_mse_pred = score_all_cells_mse_save(
            run_id=run_id,
            run_dir=run_dir,
            client=client,
            all_expr_file=expr_file,
            thresh_mix_file=thresh_mix_file,
            thresh_mix_meta=thresh_mix_meta,
            AE=AE,
            old_CV=old_CV,
            do_transform=transform_syn_mix,
        )

    meta["mse"] = sc_mse
    meta["Prediction"] = sc_mse_pred

    try:
        sc_enc = np.load(f"{run_dir}/NT_encoded.npy")
        sc_y1 = np.load(f"{run_dir}/NT_Y1.npy")
    except FileNotFoundError:
        sc_enc, sc_y1 = get_and_save_enc(run_id, run_dir, client, expr_file, AE, old_CV)
    meta[["UMAP1", "UMAP2"]] = sc_y1

    n_latent = sc_enc.shape[1]
    latent_cols = [f"enc{i+1}" for i in range(n_latent)]

    meta[latent_cols] = sc_enc

    if inc_rel_error:
        try:
            sc_rel_err = np.load(f"{run_dir}/pred_rel_error.npy")
            sc_re_pred = np.load(f"{run_dir}/pred_re_pred.npy")
        except FileNotFoundError:
            sc_rel_err, sc_re_pred = score_all_cells_relative_save(
                run_id=run_id,
                run_dir=run_dir,
                client=client,
                all_expr_file=expr_file,
                thresh_mix_file=thresh_mix_file,
                thresh_mix_meta=thresh_mix_meta,
                AE=AE,
                old_CV=old_CV,
                do_transform=transform_syn_mix,
            )

        meta["RE_prediction"] = sc_re_pred
        meta["rel_err"] = sc_rel_err

    return meta


def score_all_cells_relative_save(
    run_id,
    run_dir,
    client,
    all_expr_file,
    thresh_mix_file,
    thresh_mix_meta,
    AE,
    old_CV,
    eps: float = 1e-5,
    do_transform: bool = False,
    threshold={"tpr": 0.8},
):

    model, scaler = find_and_load_checkpoint(run_id, run_dir, client, AE, old_CV)
    thres = find_treshold_re(
        model=model,
        thresh_mix_file=thresh_mix_file,
        thresh_mix_meta=thresh_mix_meta,
        scaler=scaler,
        eps=eps,
        do_transform=do_transform,
        threshold=threshold,
    )

    all_expr_raw = pd.read_csv(all_expr_file, index_col=0)
    if do_transform:
        t2 = asinh_transform(all_expr_raw, get_co_factors())
        all_expr = pd.DataFrame(scaler.transform(t2), columns=all_expr_raw.columns)
    else:
        all_expr = pd.DataFrame(
            scaler.transform(all_expr_raw), columns=all_expr_raw.columns
        )

    with torch.no_grad():
        marker_pred = model.predict(torch.from_numpy(np.array(all_expr))).numpy()
    marker_error = (abs(all_expr - marker_pred)) / (all_expr + eps)
    error = marker_error.mean(axis=1)
    pred_bool = error > thres
    pred = ["Blast" if bool_ else "Healthy" for bool_ in pred_bool]

    np.save(f"{run_dir}/pred_rel_error.npy", error)
    np.save(f"{run_dir}/pred_re_pred.npy", pred)

    return error, pred


def score_all_cells_mse_save(
    run_id,
    run_dir,
    client,
    all_expr_file,
    thresh_mix_file,
    thresh_mix_meta,
    AE,
    old_CV,
    do_transform: bool = False,
    save: bool = True,
    threshold={"tpr": 0.8},
    model=None,
    scaler=None,
):

    if model is None and scaler is None:
        model, scaler = find_and_load_checkpoint(run_id, run_dir, client, AE, old_CV)

    thres = find_treshold_mse(
        model=model,
        thresh_mix_file=thresh_mix_file,
        thresh_mix_meta=thresh_mix_meta,
        scaler=scaler,
        do_transform=do_transform,
        threshold=threshold,
    )

    all_expr_raw = pd.read_csv(all_expr_file, index_col=0)
    if do_transform:
        t2 = asinh_transform(all_expr_raw, get_co_factors())
        all_expr = pd.DataFrame(scaler.transform(t2), columns=all_expr_raw.columns)
    else:
        all_expr = pd.DataFrame(
            scaler.transform(all_expr_raw), columns=all_expr_raw.columns
        )

    with torch.no_grad():
        marker_pred = model.predict(torch.from_numpy(np.array(all_expr))).numpy()
    marker_error = (np.array(all_expr) - marker_pred) ** 2
    error = marker_error.mean(axis=1)
    pred_bool = error > thres
    pred = ["Blast" if bool_ else "Healthy" for bool_ in pred_bool]

    if save:
        np.save(f"{run_dir}/NT_single_cell_mse.npy", error)
        np.save(f"{run_dir}/NT_single_cell_hb_pred.npy", pred)

    return error, pred


def get_and_save_enc(
    run_id, run_dir, client, all_expr_file, AE, old_CV, do_transform: bool = False
):

    model, scaler = find_and_load_checkpoint(run_id, run_dir, client, AE, old_CV)

    all_expr_raw = pd.read_csv(all_expr_file, index_col=0)
    if do_transform:
        t2 = asinh_transform(all_expr_raw, get_co_factors())
        all_expr = pd.DataFrame(scaler.transform(t2), columns=all_expr_raw.columns)
    else:
        all_expr = pd.DataFrame(
            scaler.transform(all_expr_raw), columns=all_expr_raw.columns
        )

    with torch.no_grad():
        enc = model.endec.get_latent_space(torch.from_numpy(np.array(all_expr))).numpy()

    print("Doing UMAP, this is slow")
    umapper = umap.UMAP()
    y1 = umapper.fit_transform(enc)

    np.save(f"{run_dir}/NT_encoded.npy", enc)
    np.save(f"{run_dir}/NT_Y1.npy", y1)

    return enc, y1

    # Threshold finding


def find_treshold_mse(
    model, thresh_mix_file: str, thresh_mix_meta: str, scaler, do_transform, threshold
):
    # Find threshold
    expr_name = thresh_mix_file
    meta_name = thresh_mix_meta

    mix = pd.read_csv(expr_name, index_col=0)
    mix_meta = pd.read_csv(meta_name, index_col=0)
    mix_meta["Truth"] = np.array(
        [1 if celltype == "Blast" else 0 for celltype in mix_meta["cell_type"]]
    )

    if do_transform:
        temp = asinh_transform(mix, get_co_factors())
        expr = pd.DataFrame(scaler.transform(temp), columns=temp.columns)
    else:
        expr = pd.DataFrame(scaler.transform(mix), columns=mix.columns)

    with torch.no_grad():
        marker_pred = model.predict(torch.from_numpy(np.array(expr))).numpy()

    error = ((np.array(expr) - marker_pred) ** 2).mean(axis=1)
    fpr, tpr, thresholds = roc_curve(mix_meta["Truth"], error)

    crit = float(list(threshold.values())[0])
    if list(threshold.keys())[0].lower() == "tpr":
        thres = thresholds[tpr > crit][0]
        sel_fpr = fpr[tpr > crit][0]
        sel_tpr = tpr[tpr > crit][0]

    elif list(threshold.keys())[0].lower() == "fpr":
        thres = thresholds[fpr > crit][0]
        sel_fpr = fpr[fpr > crit][0]
        sel_tpr = tpr[fpr > crit][0]

    print(
        f"Selection criteria: {threshold}, {crit}\n",
        f"MSE threshold: {thres}, FPR: {sel_fpr}, TPR: {sel_tpr}",
    )
    return thres


def find_treshold_re(
    model, thresh_mix_file, thresh_mix_meta, scaler, eps, do_transform, threshold
):
    # Find threshold
    mix = pd.read_csv(thresh_mix_file, index_col=0)
    mix_meta = pd.read_csv(thresh_mix_meta, index_col=0)
    mix_meta["Truth"] = np.array(
        [1 if celltype == "Blast" else 0 for celltype in mix_meta["cell_type"]]
    )

    if do_transform:
        temp = asinh_transform(mix, get_co_factors())
        expr = pd.DataFrame(scaler.transform(temp), columns=temp.columns)
    else:
        expr = pd.DataFrame(scaler.transform(mix), columns=mix.columns)

    with torch.no_grad():
        marker_pred = model.predict(torch.from_numpy(np.array(expr))).numpy()

    error = (abs(expr - marker_pred) / (expr + eps)).mean(axis=1)
    fpr, tpr, thresholds = roc_curve(mix_meta["Truth"], error)

    crit = float(list(threshold.values())[0])
    if list(threshold.keys())[0].lower() == "tpr":
        thres = thresholds[tpr > crit][0]
        sel_fpr = fpr[tpr > crit][0]
        sel_tpr = tpr[tpr > crit][0]

    elif list(threshold.keys())[0].lower() == "fpr":
        thres = thresholds[fpr > crit][0]
        sel_fpr = fpr[fpr > crit][0]
        sel_tpr = tpr[fpr > crit][0]

    print(
        f"Selection criteria: {threshold}\n",
        f"RE threshold: {thres}, FPR: {sel_fpr}, TPR: {sel_tpr}",
    )
    return thres


def best_score_to_param_setting_number(
    runs,
    group_cols: List,
    score_metric: str,
    ascending: bool = False,
    model_type: str = None,
):

    """Function without giving a parameter setting works only if the columns of the
    MLflow dataframe are not renamed/simplyfied!!!!"""

    avg_runs = runs.groupby(group_cols, dropna=False).mean().reset_index(drop=False)
    sorted_df = avg_runs.sort_values(score_metric, ascending=ascending).reset_index(
        drop=True
    )

    if model_type is None:
        best_run = sorted_df.iloc[0, :]

    else:
        best_run = (
            sorted_df[sorted_df["models.model"] == model_type]
            .reset_index(drop=True)
            .iloc[0, :]
        )

    if "param_setting" in group_cols:
        param_setting = best_run["param_setting"]

    else:
        sel_model_params = {
            k: v
            for k, v in best_run.to_dict().items()
            if (k.startswith("models.")) & ~(isinstance(v, float))
        }

        param_setting = runs[
            (runs[list(sel_model_params)] == pd.Series(sel_model_params)).all(axis=1)
        ]["param_setting"].unique()[0]

    return param_setting
