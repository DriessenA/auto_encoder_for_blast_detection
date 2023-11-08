import pandas as pd
from scipy.spatial.distance import pdist, squareform


def patient_diag_relapse_distance(data, run_meta, patient, linkage_method, dist_metric):
    temp = data.copy()
    temp[["patient_id", "time_point", "Prediction"]] = run_meta[
        ["patient_id", "time_point", "Prediction"]
    ]
    temp = temp.loc[
        (temp["Prediction"] == "Blast") & (temp["time_point"] != "EOI I"),
        temp.columns != "Prediction",
    ]

    patient_data = temp[
        (temp["patient_id"] == patient) & (temp.isna().sum(axis=1) == 0)
    ]

    if "Diagnosis" in list(patient_data["time_point"].unique()) and "Relapse" in list(
        patient_data["time_point"].unique()
    ):

        if linkage_method == "centroid":
            dist = squareform(
                pdist((patient_data.groupby("time_point").mean()), metric=dist_metric)
            )
            diag_relapse_dist = dist.max()

            return diag_relapse_dist

        dist = squareform(pdist(patient_data[data.columns], metric=dist_metric))
        dist = pd.DataFrame(
            dist, index=patient_data["time_point"], columns=patient_data["time_point"]
        )
        dist = dist.loc[dist.index == "Diagnosis", dist.columns == "Relapse"]

        if linkage_method == "complete" or linkage_method == "maximum":
            diag_relapse_dist = dist.max().max()
        elif linkage_method == "single":
            diag_relapse_dist = dist.min().min()
        elif linkage_method == "average":
            diag_relapse_dist = dist.mean().mean()
        else:
            Exception(
                """Please choose suitable linkage method: """
                """centroid, complete, single, average"""
            )

        return diag_relapse_dist

    else:
        print(f"Patient {patient} doesn't have Diagnosis and Relapse sample")
