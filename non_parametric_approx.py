import argparse
import numpy as np
import util
import time
import matplotlib.pyplot as plt
import matplotlib.collections as mc
import json
import os


def find_non_parametric_labels(W, labels, num_labeled, Lambda, inv_function):
    """find labels for subset U of overall unlabeled set in non_parametric_approx

    Args:
        W (ndarray): W_UU weight matrix subset
        labels (ndarray): overall labels array
        num_labeled (int): number of labels to use
        Lambda (int): hyperparameter
        inv_function (function): function to use for inverse

    Returns:
        _type_: _description_
    """
    labels[num_labeled:] = np.zeros((labels.shape[0] - num_labeled))
    n = W.shape[0]
    delta_L = np.zeros(W.shape)
    delta_L[0:num_labeled, 0:num_labeled] = np.identity(num_labeled)

    Diag = np.diag(np.sum(W, axis=1))

    A = Lambda * delta_L + Diag - W
    b = Lambda * labels
    solution = inv_function(A, b)
    return solution


def extend_non_parametric_labels(labeled_points, labels, unlabeled_points, sigma):
    """extend labels from set L \cup U to larger set

    Args:
        labeled_points (ndarray): labeled points
        labels (ndarray ): labels for the labeled points
        unlabeled_points (ndarray): points to label
        sigma (int): parameter sigma

    Returns:
        ndarray: predictions for unlabeled points
    """
    n = labeled_points.shape[0]
    N = unlabeled_points.shape[0]
    distances = np.zeros((n, N))
    for row in range(N):
        distances[:, row] = np.exp(
            -np.linalg.norm(labeled_points - unlabeled_points[row], axis=1) ** 2 / (sigma**2)
        )

    predictions = np.sum(distances * np.expand_dims(labels, axis=1), axis=0) / np.sum(
        distances, axis=0
    )
    return predictions


def transform_matrix(matrix, kNN=None, sigma=1):
    """apply gaussian kernal/kNN to matrix"""
    n = matrix.shape[0]
    # kernal
    matrix = np.exp(-(matrix**2 / sigma**2))
    # kNN
    if kNN is not None:
        matrix = util.k_highest_neighbors(matrix, kNN)
    return matrix


def non_parametric_approx(
    W, y, main_points, extended_points, num_labeled, Lambda, sigma, inv_function
):
    """solves a single instance of the non parametric approximation problem, finding labels for some set L and extending them to some larger set

    Args:
        W (ndarray): weight matrix for subset L \cup U
        y (ndarray): labels for L
        main_points (ndarray): set of points in L \cup U
        extended_points (ndarray): set of points to extend
        num_labeled (int): number of labeled examples
        Lambda (int): hyperparameter
        sigma (int): hyperparameter for gaussian
        inv_function (function): function to be used when inverting

    Returns:
        (int, ndarray, int, ndarray, int): return accuracy and incorrect values for main and extended points, as well as overall time
    """
    subset_size, extrapolate_size = W.shape[0], extended_points.shape[0]
    time1 = time.time()

    y_main, y_extended = y[0:subset_size], y[subset_size : subset_size + extrapolate_size]
    solution_main = find_non_parametric_labels(
        W, np.copy(y_main), num_labeled, Lambda, inv_function
    )

    extended_solution = extend_non_parametric_labels(
        main_points, solution_main, extended_points, sigma
    )

    time2 = time.time()

    accuracy_main, incorrect_main = util.measure_accuracy(solution_main, y_main, 0.5)
    accuracy_extended, incorrect_extended = util.measure_accuracy(
        extended_solution, y_extended, 0.5
    )

    return accuracy_main, incorrect_main, accuracy_extended, incorrect_extended, time2 - time1


def approx_feedback_set(
    W,
    Y_l,
    main_points,
    extended_points,
    sigma_0,
    Lambda,
    inv_function,
    eps=1e-3,
    delta=1,
    unlabeled_subset=None,
    kNN=None,
):
    """given a weight matrix W and labels Y and a starting sigma value, find a piecewise constant interval for sigma_0 with tolerance epsilon

    Args:
        W (ndarray): weight matrix
        Y_l (ndarray): labels
        sigma_0 (int): initial sigma value
        eps (int): tolerance for stopping for change in either sigma and function values
        delta (int): step size for nesterov's accelerated gradient descent
        num_labeled (int): number of labeled examples
        inv_function (function): inverse function to use
        unlabeled_subset (list, optional): number of unlabeled points to check. used mainly for debugging.
        Defaults to None, causing all unlabeled points to be checked.

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    num_labeled = Y_l.shape[0]
    num_unlabeled = extended_points.shape[0]
    n, N = main_points.shape[0], extended_points.shape[0]
    if unlabeled_subset is None:
        unlabeled_subset = range(num_unlabeled)
    max_sigma, min_sigma = float("inf"), -float("inf")
    overallcount = 0

    for x in unlabeled_subset:
        sigma_diff = float("inf")
        sigma = sigma_0
        old_sigma = -1
        count = 0
        # used to calculate difference in f_u values
        old_f_u = 10
        f_u = np.ones((num_unlabeled))
        # constants for nesterov's accelerated gradient descsent
        lmda_s = 1
        gamma = 0

        dist_x = np.linalg.norm(main_points - extended_points[x], axis=1)
        while sigma < max_sigma and sigma > min_sigma:
            count += 1
            if count > 100:
                break

            # perform regular algorithm to retreive unlabeled point predictions

            W_sigma = np.exp(-(W**2 / sigma**2))
            if kNN is not None:
                W_sigma = util.k_highest_neighbors(W_sigma, kNN)
            Y_full = np.zeros(n)
            Y_full[0:num_labeled] = Y_l
            n = W.shape[0]
            delta_L = np.zeros(W.shape)
            delta_L[0:num_labeled, 0:num_labeled] = np.identity(num_labeled)

            Diag = np.diag(np.sum(W_sigma, axis=1))

            A = Lambda * delta_L + Diag - W_sigma
            b = Lambda * Y_full

            try:
                f_l = inv_function(A, b)
            except:
                # print("Singular matrix")
                break

            W_x = np.exp(-(dist_x**2) / (sigma**2))
            total_distance = np.sum(W_x, axis=0)

            f_u = np.sum(W_x * f_l, axis=0) / total_distance

            function_diff = abs(f_u - old_f_u)

            if function_diff < eps and sigma_diff < eps:
                # print("breaking, didn't find point")
                break
            else:
                old_f_u = f_u

            g_u = (f_u - 0.5) ** 2
            dW_dsigma = 2 * W_sigma * W**2 / sigma**3
            dA_dsigma = np.diag(np.sum(dW_dsigma, axis=1)) - dW_dsigma

            try:
                df_dsigma = -inv_function(dA_dsigma, dA_dsigma @ f_l)
            except:
                print(f"Singular matrix")
                break

            dW_x_dsigma = 2 * W_x * dist_x**2 / sigma**3
            d_tilde_f_dsigma = (
                np.sum(dW_x_dsigma * f_l) + np.sum(W_x * df_dsigma) + f_u * np.sum(dW_x_dsigma)
            ) / total_distance

            dgu_dsigma = 2 * (f_u - 0.5) * d_tilde_f_dsigma

            # step closer to our solution
            Nesterov = delta * dgu_dsigma
            Newton = float("inf")
            if dgu_dsigma != 0:
                Newton = g_u / dgu_dsigma

            if abs(Nesterov) < abs(Newton):
                new_sigma = sigma - Nesterov
                sigma_diff = abs(sigma - new_sigma)
                old_sigma = sigma

                lmda_s_1 = (1 + (1 + 4 * lmda_s**2) ** 0.5) / (2)
                gamma = (1 - lmda_s) / lmda_s_1
                sigma = (1 - gamma) * new_sigma + gamma * sigma
                lmda_s = lmda_s_1
            else:
                new_sigma = sigma - Newton
                sigma_diff = abs(sigma - new_sigma)

                sigma = new_sigma

        overallcount += count

        # update interval if necessary
        if g_u < eps:
            if sigma > sigma_0:
                max_sigma = min(max_sigma, sigma)
            else:
                min_sigma = max(min_sigma, sigma)
    return min_sigma, max_sigma, overallcount


def find_accuracy_values(
    sigma_0,
    sigma_end,
    Lambda,
    W,
    Y,
    main_points,
    extended_points,
    num_labeled,
    inv_function,
    problem_instance_function,
    step_size=0.05,
    kNN=None,
):
    """find accuracy over some sigma range with some step size for a given problem (W, Y, num_labeled).
    Also takes in an inverse function (full inverse vs CG method), problem instance function (single vs multiclass classification), and an accuracy function

    Args:
        sigma_0 (int): sigma start value
        sigma_end (int): sigma end value
        W (ndarray): weight matrix before sigma transformation
        Y (ndarray): all labels
        num_labeled (int): number to label
        inv_function (function): inverse function to use
        problem_instance_function (function): type of classificationn function to use
        accuracy_function (function): accuracy function (differs for single vs multi-class classification)
        step_size (float, optional): step size for sigma start to sigma end. Defaults to .05.
        kNN (int, optinoal): if not None, use kNN graphs for calculation with set neighbors

    Returns:
        ndarray: array of accuracy values over sigma start to sigma end
    """
    val_range = np.arange(sigma_0, sigma_end, step_size)
    arr_main, arr_extended, incorrect_extended_list = (
        np.zeros(val_range.shape),
        np.zeros(val_range.shape),
        [],
    )

    # measure accuracy for all sigma in val range
    # time1 = time.time()
    for index, sigma in enumerate(val_range):
        sigma_matrix_array_inverse = transform_matrix(np.copy(W), kNN, sigma)
        (
            accuracy_main,
            incorrect_main,
            accuracy_extended,
            incorrect_extended,
            time,
        ) = problem_instance_function(
            sigma_matrix_array_inverse,
            Y,
            main_points,
            extended_points,
            num_labeled,
            Lambda,
            sigma,
            inv_function,
        )
        arr_main[index], arr_extended[index] = accuracy_main, accuracy_extended
        incorrect_extended_list.append(incorrect_extended.tolist())
    return arr_main, arr_extended, incorrect_extended_list


def plot_all_approx_feedback_sets(
    sigma_0,
    sigma_end,
    intervals,
    ground_truth_accuracys,
    extended_accuracys,
    step_size=0.05,
    incorrect_extended=None,
    path=None,
    title=None,
    images_path=None,
    intervals_path=None,
    time=None,
    overallcount=-1,
):
    """plots intervals overlayed with accuracys from a start to end sigmma range

    Args:
        sigma_0 (int): start sigma value
        sigma_end (int): end sigma value
        intervals (list): list of tuples containing intervals
        ground_truth_accuracys (ndarray): accuracys for each sigma values
        step_size (float, optional): step size that accuracys are said to be taken at. Defaults to .05.
        path (str, optional): path to save result to. Defaults to None.
        title (str, optional): title for plot. Defaults to None.
    """
    val_range = np.arange(sigma_0, sigma_end, step_size)

    assert (
        val_range.shape == ground_truth_accuracys.shape
        and val_range.shape == extended_accuracys.shape
    ), "Accuracys should have same shape as arange(sigma_0, sigma_end, step_size), where step size defaults to .05"

    errorVals, errorsizes = [((left + right) / 2) for left, right in intervals], [
        (right - left) / 2 if right != left else 0.0001 for left, right in intervals
    ]
    y_vals = [
        0.005 + extended_accuracys[np.argmin(np.array([abs(errorval - val) for val in val_range]))]
        for errorval in errorVals
    ]
    y_vals = [y_val if y_val != float("inf") else sigma_end for y_val in y_vals]

    biglist = [
        util.errorbar(errorVals[x], y_vals[x], errorsizes[x], 0.004)
        for x in range(len(errorsizes))
        if errorVals[x] != float("inf") and errorVals[x] != -float("inf")
    ]
    lines = [elem for l in biglist for elem in l]

    lc = mc.LineCollection(lines, colors="black")
    fig, ax = plt.subplots()
    ax.add_collection(lc)
    plt.plot(val_range, ground_truth_accuracys, label="subset accuracy")
    plt.plot(val_range, extended_accuracys, label="extended accuracy")
    plt.xlabel("sigma")
    plt.ylabel("accuracy")
    plt.title(title)
    plt.legend()
    plt.savefig(images_path)
    plt.close()

    if intervals_path is not None:
        intervals_dict = {
            "ground_truth_extended": list(extended_accuracys),
            "ground_truth_subset": list(ground_truth_accuracys),
            "sigma_vals": list(val_range),
            "intervals": intervals,
            "incorrect": incorrect_extended,
            "time": time,
            "count": overallcount,
        }
        with open(intervals_path, "w") as file:
            file.write(json.dumps(intervals_dict))


def find_all_approx_feedback_sets(
    sigma_0,
    sigma_end,
    W,
    Y,
    main_points,
    extended_points,
    num_labeled,
    inv_function,
    interval_function,
    Lambda=1.4,
    step_size=0.05,
    epsilon=1e-3,
    delta=1,
    kNN=None,
    all=False,
):
    """given a starting and ending sigma, return intervals in that range

    Args:
        sigma_0 (int): start sigma value
        sigma_end (int): end sigma value
        W (ndarray): weight array before sigma
        Y (ndarray): label array
        num_labeled (int): number of labeled instances
        inv_function (function): function for inverse(i.e. full inverse or CG method)
        interval_func (function): function to use to calculate a single interval
        step_size (float, optional): step size for findind actual accuracy. Defaults to .1.
        epsilon (int, optional): paramater for interval finding algorithm. Defaults to 1e-3.
        delta (int, optional): parameter for interval finding algorithm. Defaults to 1.
        kNN (int, optional): parameter for determining kNN neighbor amount. If None, this is not used

    Returns:
        list, int: interval list with tuples for intervals, time to conduct interval algorithm
    """
    intervals = []
    sigma = sigma_0
    # create the intervals
    overallcount = 0
    time1 = time.time()
    # find all intervals
    interval_count = 0
    while sigma < sigma_end and (interval_count <= 20):
        min, maxVal, count = interval_function(
            W,
            Y[0:num_labeled],
            main_points,
            extended_points,
            sigma,
            Lambda,
            inv_function,
            epsilon,
            delta,
            kNN=kNN,
        )
        overallcount += count
        intervals.append((min, maxVal))
        # print(f"\n\n Algorithm finished, min: {min}, max: {maxVal}")
        sigma = maxVal + step_size
        interval_count += 1
    time2 = time.time()
    return intervals, time2 - time1, overallcount


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--all",
        type=bool,
        default=False,
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="MNIST",
        help="dataset to use for generation. Default is MNIST. Other options are FashionMNIST and USPS",
    )
    parser.add_argument(
        "--num_experiments",
        type=int,
        default=1,
        help="number of independent experiments to run. Defaut is 1",
    )
    parser.add_argument(
        "--PCA",
        action="store_true",
        default=False,
        help="whether or not to use PCA for cleaning data",
    )
    parser.add_argument(
        "--n_components",
        type=int,
        default=None,
        help="number of principle components to use when doing PCA",
    )
    parser.add_argument(
        "--seed", type=int, default=1, help="seed for generating subsets. Default is 1"
    )
    parser.add_argument(
        "--subset_size",
        type=int,
        default=50,
        help="subset size to use for generation. Default is 100",
    )
    parser.add_argument(
        "--extension_size",
        type=int,
        default=250,
        help="extension size for experiment. Default is 250",
    )
    parser.add_argument(
        "--num_labels", type=int, default=10, help="Number of labels for training. Default is 10"
    )
    parser.add_argument(
        "--sigma_start",
        type=float,
        default=1,
        help="sigma value to start finding intervals at. Default is 1",
    )
    parser.add_argument(
        "--sigma_end",
        type=float,
        default=4,
        help="sigma value to finish finding intervals at. Default is 4",
    )
    parser.add_argument(
        "--Lambda",
        type=float,
        default=1.4,
        help="Lambda value to to use when finding intervals. Default is 1.4",
    )
    parser.add_argument(
        "--step_size",
        type=float,
        default=0.05,
        help="sigma step size to use for calculating ground truth accuracys. Default is .05s",
    )
    parser.add_argument(
        "--inverse_type",
        type=str,
        default="CG",
        help="inverse type, should be either CG or full_inv",
    )
    parser.add_argument(
        "--CG_steps", type=int, default=None, help="number of steps in CG method. Default is 15"
    )
    parser.add_argument(
        "--kNN",
        type=int,
        default=None,
        help="k value to use kNN in accuracy and interval calculation. default is to not use kNN",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-4,
        help="epsilon value to be used in algorithm as a stopping condition. Default is 1e-4",
    )
    parser.add_argument(
        "--delta",
        type=int,
        default=1,
        help="step value to be used in algorithm for nesterov's accelerated GD. Default is 1",
    )
    parser.add_argument(
        "--images_path",
        type=str,
        default="images/",
        help="path to save images to. Default is images/",
    )
    parser.add_argument(
        "--intervals_path",
        type=str,
        default="intervals/",
        help="path to save intervals as .json files to. Default is intervals/",
    )
    args = parser.parse_args()

    if not args.PCA:
        assert args.n_components is None, "set a number of components for PCA but not using PCA"

    assert args.inverse_type in ["CG", "full_inv"], "inverse type must be either CG or full_inv"

    if args.inverse_type == "CG":
        assert args.CG_steps is not None, "need to set number of steps in CG method with --CG_steps"
        args.inv = util.cg_inv(args.CG_steps)
        args.inv_string = f"CG_{args.CG_steps}"
        args.inv_string_title = f"CG {args.CG_steps}"
    else:
        args.inv = util.basic_inv
        args.inv_string = f"full_inv"
        args.inv_string_title = f"full inv"

    args.kNN_string, args.kNN_path, args.PCA_path = "", "", ""
    if args.kNN is not None:
        args.kNN_string = f"kNN: {args.kNN}"
        args.kNN_path = f"_kNN_{args.kNN}"

    if args.PCA:
        if args.n_components is None:
            args.PCA_path = f"_PCA_86"
        else:
            args.PCA_path = f"_PCA_{args.n_components}"
    # can change labels subset to any set of two class labels for testing
    args.labels_subset = [0, 1]
    args.problem_instance_function = non_parametric_approx
    args.accuracy_function = util.measure_accuracy

    if not os.path.exists(args.images_path):
        os.mkdir(args.images_path)
    if not os.path.exists(args.intervals_path):
        os.mkdir(args.intervals_path)

    return args


def main():
    args = parse_args()
    W, Y = util.get_random_subsets(
        dataset=args.dataset,
        subset_size=args.subset_size,
        labels_subset=args.labels_subset,
        num_subsets=args.num_experiments,
        PCA=args.PCA,
        PCA_n_components=args.n_components,
        seed=args.seed,
    )
    results_X, results_Y, values_main, values_extended = util.setup_delalleau_experiment(
        dataset=args.dataset,
        subset_size=args.subset_size,
        extension_size=args.extension_size,
        labels_subset=args.labels_subset,
        num_subsets=args.num_experiments,
        PCA=args.PCA,
        PCA_n_components=args.n_components,
        seed=args.seed,
    )
    for subset_index in range(args.num_experiments):
        print(
            f"Calculating intervals, subset {subset_index + 1} of {W.shape[0]}. dataset: {args.dataset} seed {args.seed}  all: {args.all}"
        )
        curr_results_X, curr_results_Y, curr_values_main, curr_values_extended = (
            results_X[subset_index],
            results_Y[subset_index],
            values_main[subset_index],
            values_extended[subset_index],
        )

        acc_main, acc_extended, incorrect_extended = find_accuracy_values(
            sigma_0=args.sigma_start,
            sigma_end=args.sigma_end,
            Lambda=args.Lambda,
            W=curr_results_X,
            Y=curr_results_Y,
            main_points=curr_values_main,
            extended_points=curr_values_extended,
            num_labeled=args.num_labels,
            inv_function=args.inv,
            problem_instance_function=args.problem_instance_function,
            step_size=args.step_size,
            kNN=args.kNN,
        )

        intervals, time, overallcount = find_all_approx_feedback_sets(
            sigma_0=args.sigma_start,
            sigma_end=args.sigma_end,
            W=curr_results_X,
            Y=curr_results_Y,
            main_points=curr_values_main,
            extended_points=curr_values_extended,
            num_labeled=args.num_labels,
            inv_function=args.inv,
            interval_function=approx_feedback_set,
            Lambda=args.Lambda,
            step_size=args.step_size,
            epsilon=args.epsilon,
            delta=args.delta,
            kNN=args.kNN,
        )

        plot_all_approx_feedback_sets(
            sigma_0=args.sigma_start,
            sigma_end=args.sigma_end,
            intervals=intervals,
            ground_truth_accuracys=acc_main,
            extended_accuracys=acc_extended,
            step_size=args.step_size,
            incorrect_extended=incorrect_extended,
            images_path=f"{args.images_path}non_parametric_{args.dataset}_{args.extension_size}_{args.num_labels}_labels_seed_{args.seed + subset_index}_{args.inv_string}{args.kNN_path}{args.PCA_path}_{args.all}.png",
            intervals_path=f"{args.intervals_path}non_parametric_{args.dataset}_{args.extension_size}_{args.num_labels}_labels_seed_{args.seed + subset_index}_{args.inv_string}{args.kNN_path}{args.PCA_path}_{args.all}.json",
            time=time,
            overallcount=overallcount,
        )


if __name__ == "__main__":
    main()
