import argparse
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import numpy as np


def parse_argmuments():
    parser = argparse.ArgumentParser(description="File, k, n and other parameters")

    parser.add_argument(
        "-k",
        metavar="num_neighbours",
        help="Numbers of neighbours to use in calculating the posterior",
    )
    parser.add_argument(
        "-e", metavar="expression_file", help="path to file that has the expression"
    )

    parser.add_argument(
        "-f",
        metavar="file_data",
        help="path to excel file with meta data per file",
        default=None,
    )

    parser.add_argument(
        "-m",
        metavar="meta_data",
        help="path meta data with timepoint, fileID, patientID and cluster per cell",
        default=None,
    )

    parser.add_argument(
        "-l", metavar="file_labels", help="path to file labels of the single cells"
    )

    parser.add_argument(
        "-c",
        metavar="cluster_labels",
        help="path to cluster labels of the single cells",
        default=None,
    )

    parser.add_argument("-o", metavar="save_path", help="path to output directory")

    args = parser.parse_args()

    return args


def read_args(args):
    k = int(args.k)
    expression_file = args.e
    file_labels_file = args.l
    file_data_file = args.f
    cluster_labels = args.c
    cell_meta_file = args.m
    save_path = args.o

    return (
        k,
        expression_file,
        file_labels_file,
        file_data_file,
        cluster_labels,
        save_path,
        cell_meta_file,
    )


def file_to_meta_data(
    file_data: pd.DataFrame,
    file_labels: np.ndarray,
    cluster_labels: np.ndarray,
    save_path: str = None,
    save_file: bool = False,
) -> pd.DataFrame:
    meta_data = pd.DataFrame()
    meta_data["time_point"] = [
        file_data.loc[fileID, "time_point"] for fileID in file_labels
    ]
    meta_data["patient_id"] = [
        file_data.loc[fileID, "patient_id"] for fileID in file_labels
    ]
    meta_data["file_id"] = file_labels
    meta_data["cluster_id"] = cluster_labels

    if save_file:
        meta_data.to_csv(save_path + "meta_data_high_count_files.csv", index=False)

    return meta_data


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

        p(c|x_i) = \\frac{\\sum_{i \\in b_i} W_c Y_{b_i \\in c}}{\\sum_{c=1}^{C}\\sum_{i \\in b_i} W_c Y_{b_i \\in c}} \\
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


def calc_post_p(
    A: pd.DataFrame,
    variable: str,
    meta_data: pd.DataFrame,
    save_path: str = None,
    save_file: bool = False,
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

    if save_file:
        post_p_matrix.to_csv(save_path + variable + "_post_p.csv")
        class_to_class.to_csv(save_path + variable + "_classes_p.csv")

    return post_p_matrix, class_to_class


def get_NN(
    expression: pd.DataFrame,
    save_path: str = None,
    k: int = 100,
    save_file: bool = False,
) -> pd.DataFrame:
    # Train kNN classifier
    kNN = NearestNeighbors(n_neighbors=k)
    kNN.fit(expression)

    # Get the 100 nearest neighbours. NN_df is N by 100 (N cells and 100 neighbours).
    # Values are indices of the nearest neighbours
    A = kNN.kneighbors(expression, k, return_distance=False)

    if save_file:
        np.save(save_path + str(k) + "_nearest_neighbours.npy", A)

    return A


if __name__ == "__main__":
    args = parse_argmuments()
    (
        k,
        expression_file,
        file_labels_file,
        file_data_file,
        cluster_file,
        save_path,
        cell_meta_file,
    ) = read_args(args)

    expr = pd.read_csv(expression_file, index_col=0).reset_index(drop=True)

    if not cell_meta_file:
        print("No meta data of cells")
        if not file_labels_file:
            raise Exception("If not cell metadata is supplied, supply file labels")
        if not file_data_file:
            raise Exception("If not cell metadata is supplied, supply file meta data")
        if not cluster_file:
            raise Exception("If not cell metadata is supplied, supply cluster labels")
        print("Supply meta data of files, file labels and cluster labels")
        file_data = pd.read_excel(file_data_file)
        file_labels = np.load(file_labels_file)
        cluster_labels = np.load(cluster_file)

        # File data and file labels to meta_data dataframe
        meta_data = file_to_meta_data(
            file_data=file_data,
            file_labels=file_labels,
            cluster_labels=cluster_labels,
            save_path=save_path,
            save_file=False,
        )

    else:
        meta_data = pd.read_csv(cell_meta_file, index_col=0).reset_index(drop=True)

    A = get_NN(expr, save_path=save_path, k=k, save_file=True)

    for column in ["time_point", "cell_type", "file_id", "patient_id"]:
        post_p, classes = calc_post_p(
            A, variable=column, meta_data=meta_data, save_path=save_path, save_file=True
        )
