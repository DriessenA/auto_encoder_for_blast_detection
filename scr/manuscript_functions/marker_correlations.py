import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd
from typing import List
from scipy.stats import pearsonr
from statsmodels.stats.multitest import multipletests


def count_correlation_changes(
    ref_correlation, new_correlation, threshold, plot_ref: bool = False
):
    ref_thresh_corr = ref_correlation[
        (ref_correlation > threshold) | (ref_correlation < -threshold)
    ]
    ref_thresh_corr[ref_thresh_corr.isna()] = int(0)

    if plot_ref:
        sns.heatmap(ref_thresh_corr, cmap="vlag", center=0, vmin=-1, vmax=1)
        plt.show()

    ref_thresh_corr[ref_thresh_corr > 0] = int(1)
    ref_thresh_corr[ref_thresh_corr < 0] = int(-1)
    up_tri = np.triu(np.array(ref_thresh_corr))

    pos_corr = np.triu(up_tri == 1, k=1)
    neg_corr = np.triu(up_tri == -1, k=1)
    no_corr = np.triu(((up_tri != 1) & (up_tri != -1)), k=1)

    new_thresh_corr = new_correlation[
        (new_correlation > threshold) | (new_correlation < -threshold)
    ]
    new_thresh_corr[new_thresh_corr.isna()] = int(0)
    new_thresh_corr[new_thresh_corr > 0] = int(1)
    new_thresh_corr[new_thresh_corr < 0] = int(-1)
    new_up_tri = np.triu(new_thresh_corr)

    lost_corr = 0
    gained_corr = 0
    switched_corr = 0
    maintained_corr = 0

    pos_change = Counter(((up_tri - new_up_tri).astype(int))[pos_corr])
    if 0 in pos_change.keys():
        maintained_corr += pos_change[0]
    if 1 in pos_change.keys():
        lost_corr += pos_change[1]
    if 2 in pos_change.keys():
        switched_corr += pos_change[2]
    neg_change = Counter(((up_tri - new_up_tri).astype(int))[neg_corr])
    if 0 in neg_change.keys():
        maintained_corr += neg_change[0]
    if -1 in neg_change.keys():
        lost_corr += neg_change[-1]
    if -2 in neg_change.keys():
        switched_corr += neg_change[-2]
    no_change = Counter(((up_tri - new_up_tri).astype(int))[no_corr])
    if 0 in no_change.keys():
        maintained_corr += no_change[0]
    if 1 in no_change.keys():
        gained_corr += no_change[1]
    if -1 in no_change.keys():
        gained_corr += no_change[2]

    return (lost_corr, gained_corr, switched_corr, maintained_corr)


def select_data(expr, run_meta, or_selection=None, and_selection=None):
    if or_selection is not None:
        sel_str = ""
        selections = pd.Series(False, index=run_meta.index)
        for k, v in or_selection.items():
            sel = run_meta[k] == v
            selections = selections | sel
            sel_str = sel_str + f"{k}_{v}_"

    if and_selection is not None:
        selections = pd.Series(True, index=run_meta.index)
        sel_str = ""
        for k, v in and_selection.items():
            sel = run_meta[k] == v
            selections = selections & sel
            sel_str = sel_str + f"{k}_{v}_"

    selection_bool = selections
    sel_expr = expr[selection_bool]
    sel_run_meta = run_meta[selection_bool]

    return sel_expr, sel_run_meta, sel_str


def plot_marker_correlations(
    expr,
    run_meta,
    fig_dir: str,
    or_selection: dict = None,
    and_selection: dict = None,
    heatmap_order: list = None,
):

    sel_expr, sel_run_meta, sel_str = select_data(
        expr, run_meta, or_selection, and_selection
    )

    if heatmap_order is not None:
        sel_expr = sel_expr[heatmap_order]
    corr_mat = sel_expr.corr()

    # plt.figure(figsize=(12, 10))
    # g = sns.clustermap(
    #     corr_mat,
    #     cmap="vlag",
    #     dendrogram_ratio=(0.01, 0.01),
    #     cbar_pos=(0.92, 0.98, 0.05, 0.15),
    #     center=0,
    #     vmin=-1,
    #     vmax=1,
    # )
    # g.fig.suptitle(f"{sel_str}cells", y=1.05)
    # plt.savefig(
    #     f"{fig_dir}{sel_str}clustered_marker_correlation.png", bbox_inches="tight"
    # )
    # plt.tight_layout()
    # plt.show()

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_mat, cmap="vlag", center=0, vmin=-1, vmax=1)
    plt.title(f"{sel_str}cells")
    plt.savefig(f"{fig_dir}{sel_str}ordered_marker_correlation.png")
    plt.tight_layout()
    plt.show()


def count_correlation_changes_from_data(
    expr,
    run_meta,
    ref_selection,
    or_selection: dict = None,
    and_selection: dict = None,
    threshold: float = 0.5,
    plot_ref: bool = False,
    marker_order: List = None,
    significant: bool = False,
):

    ref_expr, ref_run_meta, sel_str = select_data(
        expr, run_meta, and_selection=ref_selection
    )
    sel_expr, sel_run_meta, sel_str = select_data(
        expr, run_meta, or_selection, and_selection
    )

    if marker_order is None:
        marker_order = ref_expr.columns

    ref_corr = ref_expr[marker_order].corr(method=lambda x, y: pearsonr(x, y)[0])
    new_corr = sel_expr[marker_order].corr(method=lambda x, y: pearsonr(x, y)[0])

    if significant:

        ref_p_val_mat = ref_corr.corr(method=lambda x, y: pearsonr(x, y)[1])
        ref_p_val_mat = np.triu(np.array(ref_p_val_mat), 1)
        p_vals = ref_p_val_mat[ref_p_val_mat > 0].flatten()
        reject, adj_pvals, _, _ = multipletests(p_vals, alpha=0.05, method="fdr_bh")
        ref_p_val_mat[ref_p_val_mat > 0] = adj_pvals
        til_idx = np.tril_indices_from(ref_p_val_mat, -1)
        ref_p_val_mat[til_idx] = ref_p_val_mat.T[til_idx]
        ref_p_val_mat[np.diag_indices_from(ref_p_val_mat)] = 0
        ref_corr = ref_corr[
            pd.DataFrame(
                ref_p_val_mat < 0.05, index=ref_corr.index, columns=ref_corr.columns
            )
        ]
        ref_corr[ref_corr.isna()] = 0

        new_p_val_mat = new_corr.corr(method=lambda x, y: pearsonr(x, y)[1])
        new_p_val_mat = np.triu(np.array(new_p_val_mat), 1)
        p_vals = new_p_val_mat[new_p_val_mat > 0].flatten()
        reject, adj_pvals, _, _ = multipletests(p_vals, alpha=0.05, method="fdr_bh")
        new_p_val_mat[new_p_val_mat > 0] = adj_pvals
        til_idx = np.tril_indices_from(new_p_val_mat, -1)
        new_p_val_mat[til_idx] = new_p_val_mat.T[til_idx]
        new_p_val_mat[np.diag_indices_from(new_p_val_mat)] = 0
        new_corr = new_corr[
            pd.DataFrame(
                new_p_val_mat < 0.05, index=ref_corr.index, columns=ref_corr.columns
            )
        ]
        new_corr[new_corr.isna()] = 0

    lost_corr, gained_corr, switched_corr, maintained_corr = count_correlation_changes(
        ref_corr, new_corr, threshold, plot_ref
    )

    return lost_corr, gained_corr, switched_corr, maintained_corr
