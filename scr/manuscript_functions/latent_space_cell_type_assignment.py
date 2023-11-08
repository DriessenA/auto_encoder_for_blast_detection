from sklearn.neighbors import KNeighborsClassifier
from typing import List
import numpy as np


def predict_cell_types(
    run_meta,
    non_healthy_only: bool = True,
    return_classifier: bool = False,
    healthy_tp: List[str] = ["EOI I"],
    cell_type_col: str = "cell_type",
):

    enc_cols = [col for col in run_meta.columns if col.startswith("enc")]
    h_enc = run_meta.loc[run_meta["time_point"].isin(healthy_tp), enc_cols]
    kNN = KNeighborsClassifier()
    kNN.fit(h_enc, run_meta.loc[run_meta["time_point"].isin(healthy_tp), cell_type_col])
    ct_pred = kNN.predict(
        run_meta.loc[~(run_meta["time_point"].isin(healthy_tp)), enc_cols]
    )
    print("Classes ", kNN.classes_)

    if return_classifier:
        return ct_pred, kNN
    else:
        return ct_pred


# Marker overlap functions  ######################################
# From scanpy marker_gene_overlap
# https://github.com/scverse/scanpy/blob/3d597046ac81a7f8c305cfa20594d8ecaba11fb8/scanpy/tools/_marker_gene_overlap.py#L33


def calc_overlap_count(markers1: dict, markers2: dict):
    """
    Calculate overlap count between the values of two dictionaries
    Note: dict values must be sets
    """
    overlaps = np.zeros((len(markers1), len(markers2)))

    for j, marker_group in enumerate(markers1):
        tmp = [
            len(markers2[i].intersection(markers1[marker_group]))
            for i in markers2.keys()
        ]
        overlaps[j, :] = tmp

    return overlaps


def calc_overlap_coef(markers1: dict, markers2: dict):
    """
    Calculate overlap coefficient between the values of two dictionaries
    Note: dict values must be sets
    """
    overlap_coef = np.zeros((len(markers1), len(markers2)))

    for j, marker_group in enumerate(markers1):
        tmp = [
            len(markers2[i].intersection(markers1[marker_group]))
            / max(min(len(markers2[i]), len(markers1[marker_group])), 1)
            for i in markers2.keys()
        ]
        overlap_coef[j, :] = tmp

    return overlap_coef


def calc_jaccard(markers1: dict, markers2: dict):
    """
    Calculate jaccard index between the values of two dictionaries
    Note: dict values must be sets
    """
    jacc_results = np.zeros((len(markers1), len(markers2)))

    for j, marker_group in enumerate(markers1):
        tmp = [
            len(markers2[i].intersection(markers1[marker_group]))
            / len(markers2[i].union(markers1[marker_group]))
            for i in markers2.keys()
        ]
        jacc_results[j, :] = tmp

    return jacc_results
