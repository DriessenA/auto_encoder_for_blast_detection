import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_mispredicted_blast_fraction_per_file(
    run_meta, blast_files=None, reference: str = "sample_size", cmap: dict = None
):
    sns.set_context("paper")

    if blast_files is None:
        blast_files = list(run_meta["file_id"].unique())

    # Mis predicted blast fraction
    crit = (
        (run_meta["cell_type"] == "Blast")
        & (run_meta["Prediction"] == "Healthy")
        & (run_meta["file_id"].isin(blast_files))
    )
    temp = (
        run_meta[crit]
        .groupby(["PatientID", "time_point", "file_id"])
        .size()
        .reset_index(drop=False)
        .rename(columns={0: "False Negatives"})
    )
    if reference == "sample_size":
        size_crit = run_meta["file_id"].isin(temp["file_id"])
    elif reference == "blasts":
        size_crit = (run_meta["file_id"].isin(temp["file_id"])) & (
            run_meta["Prediction"] == "Blast"
        )
    else:
        Exception("Reference should be either sample_size or blasts")

    temp2 = (
        run_meta[size_crit]
        .groupby(["PatientID", "time_point", "file_id"])
        .size()
        .reset_index(drop=False)
        .rename(columns={0: "Size"})
    )
    temp["Mispredicted blasts fraction"] = temp["False Negatives"] / temp2["Size"]
    temp = temp.sort_values(
        "PatientID", key=lambda idx: [int(x.replace("P", "")) for x in idx]
    )
    plt.figure(figsize=(15, 5))
    if cmap is None:
        sns.barplot(
            data=temp,
            x="PatientID",
            y="Mispredicted blasts fraction",
            hue="time_point",
            hue_order=sorted(temp["time_point"].unique()),
        )
    else:
        sns.barplot(
            data=temp,
            x="PatientID",
            y="Mispredicted blasts fraction",
            hue="time_point",
            palette=cmap,
            hue_order=sorted(temp["time_point"].unique()),
        )

    return plt


def get_blast_prop(
    run_meta,
    plot: bool = False,
    full_file_size: bool = False,
    pred_col_name: str = "Prediction",
):
    sns.set_context("paper")

    file_bh_counts = (
        run_meta.groupby(["file_id", pred_col_name])
        .size()
        .reset_index(drop=False)
        .rename(columns={0: "counts"})
    )

    if not full_file_size:
        file_size = (
            run_meta.groupby(["file_id", "time_point", "PatientID"])
            .size()
            .reset_index(drop=False)
            .rename(columns={0: "size"})
        )

    else:
        file_size = run_meta[["file_id", "time_point", "PatientID"]].drop_duplicates()
        file_size["size"] = 10000

    file_proportions = pd.merge(file_size, file_bh_counts, on="file_id")
    file_proportions["prop"] = file_proportions["counts"] / file_proportions["size"]

    if plot:
        file_proportions.sort_values(
            "PatientID", key=lambda idx: [int(x.replace("P", "")) for x in idx]
        )
        plt.figure(dpi=100)
        sns.barplot(data=file_proportions, x="time_point", y="prop", hue=pred_col_name)
        plt.show()

        plt.figure(dpi=100)
        sns.barplot(
            data=file_proportions[file_proportions[pred_col_name] == "Blast"],
            x="time_point",
            y="prop",
        )
        plt.show()

        plt.figure(figsize=(18, 10), dpi=100)
        sns.barplot(
            data=file_proportions[file_proportions[pred_col_name] == "Blast"],
            x="PatientID",
            y="prop",
            hue="time_point",
        )
        plt.show()

    return file_proportions
