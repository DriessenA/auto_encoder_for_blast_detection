import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np

sns.set_context("paper")


def plot_bw_heatmap(
    expression: pd.DataFrame,
    meta_data,
    group_var: str = "cluster_id",
    fig_size: tuple = (15, 10),
    scale=None,
    row_colors: pd.Series = None,
    cmap: dict = None,
):
    bwg_cmap = ListedColormap(["white", "lightgrey", "grey", "black"])
    row_cluster = True

    plot_data = expression.copy()
    plot_data[group_var] = meta_data[group_var].values
    plot_data = plot_data.groupby([group_var]).median()

    if row_colors is not None:
        row_cluster = False

        if cmap is None:
            lut = dict(
                zip(
                    row_colors.unique(),
                    sns.color_palette("husl", len(row_colors.unique())),
                )
            )
        else:
            lut = cmap
        row_colors = row_colors.map(lut)

    sns.clustermap(
        plot_data,
        metric="correlation",
        method="average",
        standard_scale=None,
        cmap=bwg_cmap,
        figsize=fig_size,
        linewidths=0.5,
        linecolor="lightgray",
        col_cluster=False,
        row_cluster=row_cluster,
        row_colors=row_colors,
    )

    if row_colors is not None:
        handles = [Patch(facecolor=lut[name]) for name in lut]
        plt.legend(
            handles,
            lut,
            title="",
            bbox_to_anchor=(1, 1),
            bbox_transform=plt.gcf().transFigure,
            loc="upper right",
        )

    return plt


def plot_all_markers(Y1, data_norm, sel_markers, cmap="viridis", save_fig=False):
    plt.figure(figsize=(30, 30))
    for m in range(len(sel_markers)):
        plt.subplot(5, 5, m + 1)
        plt.scatter(Y1[:, 0], Y1[:, 1], s=0.5, c=data_norm.iloc[:, m], cmap=cmap)
        plt.title(sel_markers[m])
        # plt.xticks([])
        # plt.yticks([])
    if save_fig:
        plt.savefig("UMAP_markers.pdf", dpi=300)

    return plt


def UMAP_annotation_vs_rest(
    Y1,
    annotation,
    highlight: str,
    downlight_name: str = "Rest",
    dpi: int = 300,
    s=0.5,
    cmap: dict = None,
):
    highlight_Y1 = Y1[annotation == highlight]
    downlight_Y1 = Y1[annotation != highlight]

    plot_data = np.vstack([downlight_Y1, highlight_Y1])

    col_vec = [downlight_name] * downlight_Y1.shape[0] + [
        highlight
    ] * highlight_Y1.shape[0]

    plt.figure(figsize=(10, 10), dpi=dpi)
    if cmap is None:
        sns.scatterplot(
            x=plot_data[:, 0], y=plot_data[:, 1], hue=col_vec, alpha=0.6, s=0.5
        )
    else:
        sns.scatterplot(
            x=plot_data[:, 0],
            y=plot_data[:, 1],
            hue=col_vec,
            alpha=0.6,
            s=0.5,
            palette=cmap,
        )
    return plt
