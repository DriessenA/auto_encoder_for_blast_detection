import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from .marker_correlations import select_data
from typing import Optional, Literal
from sklearn.mixture import GaussianMixture


def calc_post_p(
    A: pd.DataFrame, variable: str, meta_data: pd.DataFrame
) -> pd.DataFrame:
    """Calculate the posterior probability that a cell comes from class
        of the given variable given the class of its k nearest neighbours.
    See respective methods for how the posterior is calculated

    The class to class probabilities are calculates by averaging over
    the posterior probabilities of all cells that the same true class label.

    ..math::

        \frac{\\sum_{x_i|Y_i=c}p(c|x_i)}{n_{Y_i=c}}

    Arguments:
        A: Dataframe containing for each cell (rows) the k-nearest neighbours indices.
            The indices locate the corresponding meta data of the cells in the
            `meta_data`.
        variable: variable for which to calculate the posterior class probabilities.
        meta_data: Contains the variable of interest. Is sliced with the indices given
            by a row in A to get varaible classes of the nearest neighbours.

    Returns:
        post_p_matrix: N by C matrix, where C is the number of classes in the given
            variable. Where entry `post_p_matrix`_ij is the posterior probability
            that cell i came from class j given its k-nearest neigbours. Also one
            column with name `pred_class` is the class assigned to the cell based on
            the highest posterior probability.
        class_to_class: C by C matrix, so a class to class probability where the
            diagonal elements indicate how self-contained a class is. Obtained by
            averagering all posterior probabilities of cells that are assigned
            the same class.
    """

    NN_df = pd.DataFrame(A)

    # Get the posterior probability for each cell for each class and class assignements
    post_p_matrix = NN_df.apply(
        calc_inverse_weights, args=(meta_data, variable), axis=1
    )

    # Average over all cells with the same class
    post_p_matrix["class"] = meta_data[variable]
    temp = post_p_matrix.drop("pred_class", inplace=False, axis=1)
    class_to_class = temp.groupby("class").mean()

    # Some classes do not show up in the rows of the class to class matrix
    # Because this class is not pedicted for any cell
    #  (not the highest posterior prob. for any cell)
    # So here we add these classes as rows all probabilities zero to get a square matrix
    missing_classes = (
        (set(post_p_matrix.columns))
        - set(class_to_class.index)
        - {"class", "pred_class"}
    )
    for class_name in missing_classes:
        class_to_class = class_to_class.append(
            pd.Series(
                [0] * class_to_class.shape[1],
                name=class_name,
                index=class_to_class.columns,
            )
        )

    # Order columns same as rows so diagonal elements
    # are the self-containment of the class
    class_to_class = class_to_class[class_to_class.index]

    return post_p_matrix, class_to_class


def calc_inverse_weights(
    NN_row: np.ndarray, meta_data: pd.DataFrame, variable: str
) -> pd.Series:
    """ This function calculates the posterior probability that a cell
    belongs to a class of the given variable, based on the classes of
    its k nearest neighbours. The neighbor counts are weighted
    by the inverse of how often this class occurs among all cells.
    This function operates on a 1-dimensional array of length k,
    which can be a row of a larger dataframe.

    The posterior probability that a cell :math: `i` with expression :math: `x_i`
    orignates from class :math: `c`, depending on it nearest neighbours
     :math: `b_i` is calculated as follows:
      .. math::

        p(c|x_i) = \\frac{\\sum_{i \\in b_i} W_c Y_{b_i
        \\in c}}{\\sum_{c=1}^{C}\\sum_{i \\in b_i} W_c Y_{b_i \\in c}} \\
        W_c = \\frac{1}{\\sum_{n=1}^{N} Y_{n\\in c}}

    Where :math: `W_c` is the prior probability of class :math: `c`
    and :math: `Y_i` are the class labels of the nearest neighbours.
    And :math: `Y` equals :math:`1` if neighbour :math: `b_i` or cell :math: `n`
    has class :math: `c` and :math: `0` otherwise.

    Arguments:
        NN_row: array of length k, with k being the number of nearest neighbours.
            Values are the indices of the nearest neighbours in the meta_data
        meta_data: Contains the variable of interest. Is sliced with
            the indices given by NN_row to get varaible classes
            of the nearest neighbours.
        variable: variable for which to calculate the posterior class probabilities.

    Returns: A series with for each class the posterior probabiltiy that
        the cell came from that class and a `pEed_class` entry that indicated
        the class assigned to the cell based on the maximal posterior probability.
    """

    # Get classes of the neighbours
    NN_vars = meta_data.loc[NN_row, variable]

    # Count how often classes occur among neighbours
    NN_vars_counts = NN_vars.value_counts()

    # For classes that don't occur the count is 0,
    # ensure we have vector the contains all classes
    NN_v_counts = pd.Series(0, index=meta_data[variable].unique())
    NN_v_counts[NN_vars_counts.index] = NN_vars_counts

    # Count how often class occurs among all cells
    # and take the inverse of that as priors
    Wc = meta_data.groupby(variable).count().iloc[:, 0]
    Wc = 1 / Wc

    # Calculate posterior probability and add an entry
    # for the class assignement based on the maximum posterior prob.
    post_p = NN_v_counts * Wc / sum(NN_v_counts * Wc)
    post_p["pred_class"] = post_p.idxmax()

    return post_p


def get_NN(
    expression: pd.DataFrame,
    k: int = 100,
) -> pd.DataFrame:
    # Train kNN classifier
    kNN = NearestNeighbors(n_neighbors=k)
    kNN.fit(expression)

    # Get the 100 nearest neighbours. NN_df is N by 100 (N cells and 100 neighbours).
    # Values are indices of the nearest neighbours
    A = kNN.kneighbors(expression, k, return_distance=False)

    return A


def get_predict_proba(
    expr,
    run_meta,
    selection1,
    selection2,
    variable,
    model: Literal["individuality", "GMM"],
    model_args: Optional[dict] = None,
    return_model: bool = False,
):

    if selection1["time_point"] == "Diagnosis":
        sel_expr1, sel_run_meta1, sel_str1 = select_data(
            expr=expr, run_meta=run_meta, and_selection=selection1
        )
        sel_expr2, sel_run_meta2, sel_str2 = select_data(
            expr=expr, run_meta=run_meta, and_selection=selection2
        )
    elif selection2["time_point"] == "Diagnosis":
        sel_expr1, sel_run_meta1, sel_str1 = select_data(
            expr=expr, run_meta=run_meta, and_selection=selection2
        )
        sel_expr2, sel_run_meta2, sel_str2 = select_data(
            expr=expr, run_meta=run_meta, and_selection=selection1
        )
    else:
        sel_expr1, sel_run_meta1, sel_str1 = select_data(
            expr=expr, run_meta=run_meta, and_selection=selection1
        )
        sel_expr2, sel_run_meta2, sel_str2 = select_data(
            expr=expr, run_meta=run_meta, and_selection=selection2
        )

    sel_expr = pd.concat([sel_expr1, sel_expr2]).reset_index(drop=True)
    sel_run_meta = pd.concat([sel_run_meta1, sel_run_meta2]).reset_index(drop=True)

    if model == "individuality":
        A = get_NN(sel_expr, **model_args)

        post_p, classes = calc_post_p(A, variable=variable, meta_data=sel_run_meta)

    elif model == "GMM":
        post_p, classes, model = get_GMM_proba(
            sel_expr=sel_expr,
            sel_run_meta=sel_run_meta,
            variable=variable,
            model_args=model_args,
        )

    else:
        post_p, classes = get_predict_proba_model(
            model, sel_expr, sel_run_meta, variable
        )

    if return_model:
        return post_p, classes, model

    else:
        return post_p, classes


def get_predict_proba_model(model, expr, meta, variable):

    model.fit(expr, meta[variable])
    proba = model.predict_proba(expr)

    pred = pd.DataFrame(proba, columns=meta[variable].unique())
    pred["pred_class"] = model.predict(expr)
    pred["class"] = meta[variable]

    temp = pred.drop("pred_class", inplace=False, axis=1)
    class_to_class = temp.groupby("class").mean()

    # Some classes do not show up in the rows of the class to class matrix
    # Because this class is not pedicted for any cell
    #  (not the highest posterior prob. for any cell)
    # So here we add these classes as rows all probabilities zero to get a square matrix
    missing_classes = (
        (set(pred.columns)) - set(class_to_class.index) - {"class", "pred_class"}
    )
    for class_name in missing_classes:
        class_to_class = class_to_class.append(
            pd.Series(
                [0] * class_to_class.shape[1],
                name=class_name,
                index=class_to_class.columns,
            )
        )

    # Order columns same as rows so diagonal elements
    # are the self-containment of the class
    class_to_class = class_to_class[class_to_class.index]

    return pred, class_to_class


def mis_predicted_cell_to_columns_diag_relapse(post_p):
    post_p["relapse_like_diag"] = [
        "Relapse like diagnosis" if b else np.nan
        for b in (post_p["pred_class"] == "Diagnosis") & (post_p["class"] == "Relapse")
    ]
    post_p["diag_like_relapse"] = [
        "Diagnosis like relapse" if b else np.nan
        for b in (post_p["pred_class"] == "Relapse") & (post_p["class"] == "Diagnosis")
    ]

    post_p.loc[post_p["relapse_like_diag"].isnull(), "relapse_like_diag"] = post_p.loc[
        post_p["relapse_like_diag"].isnull(), "class"
    ]
    post_p.loc[post_p["diag_like_relapse"].isnull(), "diag_like_relapse"] = post_p.loc[
        post_p["diag_like_relapse"].isnull(), "class"
    ]
    post_p["diag_like_relapse"] = pd.Categorical(
        post_p["diag_like_relapse"],
        categories=["Relapse", "Diagnosis", "Diagnosis like relapse"],
        ordered=True,
    )

    post_p["relapse_like_diag"] = pd.Categorical(
        post_p["relapse_like_diag"],
        categories=["Relapse", "Diagnosis", "Relapse like diagnosis"],
        ordered=True,
    )

    return post_p


def get_GMM_proba(sel_expr, sel_run_meta, variable, model_args):

    diag_expr = sel_expr[sel_run_meta["time_point"] == "Diagnosis"]
    relapse_expr = sel_expr[sel_run_meta["time_point"] == "Relapse"]

    if model_args is not None:
        diag_gmm = GaussianMixture(**model_args)
        relapse_gmm = GaussianMixture(**model_args)
    else:
        diag_gmm = GaussianMixture(n_components=1)
        relapse_gmm = GaussianMixture(n_components=1)

    diag_gmm.fit(diag_expr)
    relapse_gmm.fit(relapse_expr)

    like_relapse = relapse_gmm.score_samples(sel_expr)
    like_diag = diag_gmm.score_samples(sel_expr)

    res = pd.DataFrame({"Diagnosis": like_diag, "Relapse": like_relapse})
    res["class"] = sel_run_meta[variable]
    res.loc[res["Diagnosis"] > res["Relapse"], "pred_class"] = "Diagnosis"
    res.loc[res["Diagnosis"] < res["Relapse"], "pred_class"] = "Relapse"

    temp = res.drop("pred_class", inplace=False, axis=1)
    class_to_class = temp.groupby("class").mean()

    # Some classes do not show up in the rows of the class to class matrix
    # Because this class is not pedicted for any cell
    #  (not the highest posterior prob. for any cell)
    # So here we add these classes as rows all probabilities zero to get a square matrix
    missing_classes = (
        (set(res.columns)) - set(class_to_class.index) - {"class", "pred_class"}
    )
    for class_name in missing_classes:
        class_to_class = class_to_class.append(
            pd.Series(
                [0] * class_to_class.shape[1],
                name=class_name,
                index=class_to_class.columns,
            )
        )

    # Order columns same as rows so diagonal elements
    # are the self-containment of the class
    class_to_class = class_to_class[class_to_class.index]

    return res, class_to_class, [diag_gmm, relapse_gmm]
