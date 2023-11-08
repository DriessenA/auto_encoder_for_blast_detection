import pandas as pd
from typing import List
import numpy as np
import ot as ot
from scipy.spatial import distance


def make_emd_cost_matrix(ref_pseudo_time, predicted_ct_name):
    dist_mat = np.zeros((len(ref_pseudo_time), len(ref_pseudo_time)))
    cell_types = ref_pseudo_time[predicted_ct_name].unique()
    for i in range(len(cell_types)):
        c1 = cell_types[i]
        pt1 = float(
            ref_pseudo_time.loc[ref_pseudo_time[predicted_ct_name] == c1, "pseudotime"]
        )
        for j in range(i + 1, len(cell_types)):
            c2 = cell_types[j]
            pt2 = float(
                ref_pseudo_time.loc[
                    ref_pseudo_time[predicted_ct_name] == c2, "pseudotime"
                ]
            )
            dist_mat[i, j] = abs(pt1 - pt2)
            dist_mat[j, i] = abs(pt2 - pt1)

    return dist_mat


def sample_distances(
    run_meta,
    sel_patients: List = None,
    sel_timepoints: List = None,
    predicted_ct_name: str = "Predicted_cell_type",
    cell_type_order: list = [
        "HSPCs",
        "Monoblasts/Myeloblasts",
        "Pro-Monocyte",
        "Monocyte",
    ],
):

    cell_type_fraction = get_cell_type_fractions(
        run_meta,
        predicted_ct_name=predicted_ct_name,
        sel_patients=sel_patients,
        sel_timepoints=sel_timepoints,
    )

    cell_type_fraction = cell_type_fraction.set_index(
        ["patient_id", "time_point", "file_id"]
    )

    sample_pseudotime = average_sample_pseudotime(
        run_meta=run_meta,
        predicted_ct_name=predicted_ct_name,
        cell_type_order=cell_type_order,
        sel_patients=sel_patients,
        sel_timepoints=sel_timepoints,
    )

    ref_pseudo_time = calc_pseudotime(
        run_meta=run_meta,
        predicted_ct_name=predicted_ct_name,
        cell_type_order=cell_type_order,
    )

    pt_list = list(sample_pseudotime["average_pseudotime"])
    pt_diff = [abs(a - b) for a, b in zip(pt_list[::2], pt_list[1::2])]

    missing_pt_classes = set(cell_type_fraction.columns).difference(
        set(ref_pseudo_time[predicted_ct_name].unique())
    )
    average_pt = float(ref_pseudo_time.mean())
    for i, c in enumerate(missing_pt_classes):
        ref_pseudo_time = pd.concat(
            [
                pd.DataFrame(
                    {predicted_ct_name: c, "pseudotime": average_pt},
                    index=[len(ref_pseudo_time) + i],
                ),
                ref_pseudo_time,
            ]
        )
    ref_pseudo_time = ref_pseudo_time.reset_index(drop=True)

    dist_mat = make_emd_cost_matrix(ref_pseudo_time, predicted_ct_name)

    p = np.array(cell_type_fraction)
    p = p.copy(order="C")

    distances = pd.DataFrame(
        {
            "euclidean_dist": [
                distance.euclidean(p[i], p[i + 1])
                for i in range(0, len(cell_type_fraction), 2)
            ],
            "wasserstein_dist": [
                ot.emd2(p[i], p[i + 1], dist_mat)
                for i in range(0, len(cell_type_fraction), 2)
            ],
            "avg_pseudotime_dist": pt_diff,
        },
        index=cell_type_fraction.index.get_level_values("patient_id").unique(),
    )

    return distances.reset_index()


def average_sample_pseudotime(
    run_meta,
    predicted_ct_name: str = "Predicted_cell_type",
    cell_type_order: list = [
        "HSPCs",
        "Monoblasts/Myeloblasts",
        "Pro-Monocyte",
        "Monocyte",
    ],
    sel_patients: List = None,
    sel_timepoints: List = None,
    incl_ct_fractions: bool = False,
):

    """Calculate the average pseudotime for each sample based on the fraction of each
    cell type in the sample and the average pseudotime of the cell type. Celltypes
    not included in `cell_type_order` are used for calucating cell type frequencies,
    but not for the average pseudotime."""

    sel_run_meta = run_meta.copy()
    if sel_patients is not None:
        sel_run_meta = sel_run_meta[
            sel_run_meta["patient_id"].isin(sel_patients)
        ].reset_index(drop=True)
    if sel_timepoints is not None:
        sel_run_meta = sel_run_meta[
            sel_run_meta["time_point"].isin(sel_timepoints)
        ].reset_index(drop=True)

    cluster_pseudotime = calc_pseudotime(
        run_meta=run_meta,
        predicted_ct_name=predicted_ct_name,
        cell_type_order=cell_type_order,
    )

    cell_type_fractions = get_cell_type_fractions(
        run_meta=sel_run_meta, predicted_ct_name=predicted_ct_name, long_format=True
    )

    all_data = cell_type_fractions.merge(
        cluster_pseudotime, how="left", on=predicted_ct_name
    )
    all_data["average_pseudotime"] = (
        all_data["cell_type_fraction"] * all_data["pseudotime"]
    )
    avg_pseudotime = (
        all_data.groupby(["patient_id", "time_point", "file_id"])
        .sum()["average_pseudotime"]
        .reset_index(drop=False)
    )

    if not incl_ct_fractions:
        return avg_pseudotime
    else:
        pseudo_time_and_ct_freq = cell_type_fractions.merge(
            avg_pseudotime, on=["patient_id", "time_point", "file_id"], how="left"
        )

        return pseudo_time_and_ct_freq


def get_cell_type_fractions(
    run_meta,
    predicted_ct_name: str = "Predicted_cell_type",
    sel_patients: List = None,
    sel_timepoints: List = None,
    long_format: bool = False,
):
    sel_run_meta = run_meta.copy()
    if sel_patients is not None:
        sel_run_meta = sel_run_meta[
            sel_run_meta["patient_id"].isin(sel_patients)
        ].reset_index(drop=True)
    if sel_timepoints is not None:
        sel_run_meta = sel_run_meta[
            sel_run_meta["time_point"].isin(sel_timepoints)
        ].reset_index(drop=True)

    # Count cell types
    ct_counts = (
        sel_run_meta.groupby(
            ["patient_id", "time_point", "file_id", "Predicted_cell_type"]
        )
        .size()
        .reset_index(predicted_ct_name)
        .rename(columns={0: "counts"})
    )

    # Get file sizes
    file_size = sel_run_meta.groupby("file_id").size()
    file_size.name = "file_size"

    # Merge and calculate frequencies
    ct_freq = pd.merge(ct_counts, file_size, left_index=True, right_index=True)
    ct_freq["cell_type_fraction"] = ct_freq["counts"] / ct_freq["file_size"]
    ct_freq = ct_freq.fillna(0)

    if long_format:
        res = ct_freq[[predicted_ct_name, "cell_type_fraction"]].reset_index(drop=False)
        return res

    else:
        ct_freq = ct_freq.reset_index(drop=False)
        res = ct_freq[
            [
                "patient_id",
                "time_point",
                "file_id",
                predicted_ct_name,
                "cell_type_fraction",
            ]
        ].pivot(
            index=["patient_id", "time_point", "file_id"], columns=predicted_ct_name
        )
        res = res.droplevel(None, axis=1)
        res = res[ct_freq[predicted_ct_name].unique()]
        res.columns.name = ""
        res = res.fillna(0)
        res = res.reset_index(drop=False)

        return res


def calc_pseudotime(
    run_meta,
    predicted_ct_name: str = "Predicted_cell_type",
    cell_type_order: list = [
        "HSPCs",
        "Monoblasts/Myeloblasts",
        "Pro-Monocyte",
        "Monocyte",
    ],
):
    # Select only healthy reference
    sel_run_meta = run_meta[
        (run_meta["time_point"] == "EOI I")
        & (run_meta[predicted_ct_name].isin(cell_type_order))
    ]

    # Order according to developmental trajectory
    sel_run_meta.loc[:, predicted_ct_name] = sel_run_meta[predicted_ct_name].astype(
        "category"
    )
    sel_run_meta[predicted_ct_name].cat.reorder_categories(
        cell_type_order, inplace=True, ordered=True
    )
    sel_run_meta.sort_values(predicted_ct_name, inplace=True)

    # Count cell types
    healthy_reference = (
        sel_run_meta.groupby(predicted_ct_name)
        .size()
        .reset_index(predicted_ct_name)
        .rename(columns={0: "counts"})
    )
    healthy_reference["reference_fraction"] = healthy_reference["counts"] / len(
        sel_run_meta
    )
    interval_boudaries = [0] + list(healthy_reference["reference_fraction"])
    average_ct_pseudotime = [
        (a + b) / 2 for a, b in zip(interval_boudaries[:-1], interval_boudaries[1:])
    ]

    return pd.DataFrame(
        {predicted_ct_name: cell_type_order, "pseudotime": average_ct_pseudotime}
    )
