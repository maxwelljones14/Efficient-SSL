import argparse
import matplotlib.collections as mc
import matplotlib.pyplot as plt
import numpy as np
import time
import util
import scipy.sparse as sp
import json
import os

def harmonic_approx(W_sigma, Y_l, inv_function):
    """given an instance W,Y of the harmonic optimization problem and an inverse function,
       solve it

    Args:
        W (weight graph): weight graph
        Y_l (int): labels for labeled subset (assumed to be first Y_l.shape[0] entries of W)
        inv_function (function): inverse function to use

    Raises:
        ValueError: if a row is all zero, will raise error

    Returns:
        int, list, int: accuracy, error points, time taken to do inverse
    """
    num_labeled = Y_l.shape[0]
    num_unlabeled = W_sigma.shape[0] - num_labeled
    if sp.issparse(W_sigma):
        time1 = time.time()
        P = sp.csr_matrix(W_sigma)
        util.inplace_csr_row_normalize_l1(P)
        P_uu_row_normalized, P_ul_row_normalized = P[num_labeled:,num_labeled:], P[num_labeled:,0:num_labeled]
        I_minus_P_uu = sp.eye(num_unlabeled) - P_uu_row_normalized
        solution = inv_function(I_minus_P_uu, P_ul_row_normalized @ Y_l)
        time2 = time.time()
    else:
        time1 = time.time()
        W_row_sums = np.sum(W_sigma, axis=1, keepdims=True)
        P = W_sigma / W_row_sums
        if np.isnan(P).any():
            raise ValueError(f"divide by zero error when normalizing, {np.sum(P,axis = 1)}")
        P_uu_row_normalized, P_ul_row_normalized = P[num_labeled:,num_labeled:], P[num_labeled:,0:num_labeled]
        I_minus_P_uu = np.identity(num_unlabeled) - P_uu_row_normalized
        
        solution = inv_function(I_minus_P_uu, P_ul_row_normalized @ Y_l)
        time2 = time.time()

    return solution, time2 - time1

def approx_feedback_set(W, Y_l, sigma_0, inv_function, eps=1e-3, delta=1, unlabeled_subset=None, kNN=None, sparse=False):
    """given a weight matrix W and labels Y and a starting sigma value, find a piecewise constant interval for sigma_0 with tolerance epsilon 

    Args:
        W (ndarray): weight matrix
        Y_l (ndarray): labels
        sigma_0 (int): initial sigma value
        inv_function (function): inverse function to use
        eps (int): tolerance for stopping for change in either sigma and function values
        delta (int): step size for nesterov's accelerated gradient descent
        unlabeled_subset (list, optional): number of unlabeled points to check. used mainly for debugging. 
            Defaults to None, causing all unlabeled points to be checked.
        kNN (int, optional): number of neighbors for kNN graph if set, else uses full graph
        sparse (boolean): whether or not to use the scipy sparse class when doing calculations

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    num_labeled = Y_l.shape[0]
    num_unlabeled = W.shape[0] - num_labeled
    if unlabeled_subset is None:
        unlabeled_subset = range(num_unlabeled)
    max_sigma, min_sigma = float("inf"), -float("inf")
    overallcount = 0
    if sparse:
        W_low = util.k_lowest_neighbors(np.copy(W), kNN)
        sparse_neighbors = sp.coo_matrix(W_low + np.eye(W.shape[0])) #adding the identity so that the shape of the sparse neighbor graph is correct
        W_sparse = sp.coo_matrix(W_low)
    for x in unlabeled_subset:
        sigma_diff = float('inf')
        sigma = sigma_0
        old_sigma = -1
        # if > 100 iterations, break
        count = 0
        # used to calculate difference in f_u values
        old_f_u = np.ones((num_unlabeled)) * 10
        f_u = np.ones((num_unlabeled))
        # constants for nesterov's accelerated gradient descsent
        lmda_s = 1
        gamma = 0
        g_u, dgu_dsigma = np.zeros(num_unlabeled), np.zeros(num_unlabeled)
        while sigma < max_sigma and sigma > min_sigma:
            count += 1
            if count > 100:
                break

            # perform regular algorithm to retreive unlabeled point predictions

            if sparse:
                sparse_neighbors_sigma = (-(sparse_neighbors.multiply(sparse_neighbors) / sigma ** 2)).expm1()
                sparse_neighbors_sigma[sparse_neighbors_sigma.nonzero()]  += 1
                sparse_neighbors_sigma.setdiag([1 for _ in range(W.shape[0])])
                W_sigma = sparse_neighbors_sigma

                P_sparse = sp.csr_matrix(W_sigma)
                util.inplace_csr_row_normalize_l1(P_sparse)
                P_uu_row_normalized, P_ul_row_normalized = P_sparse[num_labeled:,num_labeled:], P_sparse[num_labeled:,0:num_labeled]
                I_minus_P_uu = sp.eye(num_unlabeled) - P_uu_row_normalized
            else:
                W_sigma = np.exp(-(W ** 2 / sigma ** 2))
                if kNN is not None:
                    W_sigma = util.k_highest_neighbors(W_sigma, kNN)


                W_row_sums = np.sum(W_sigma, axis=1, keepdims=True)
                P = W_sigma / W_row_sums
                if np.isnan(P).any():
                    raise ValueError(f"divide by zero error when normalizing, {np.sum(P,axis = 1)}")
                P_uu_row_normalized, P_ul_row_normalized = P[num_labeled:,num_labeled:], P[num_labeled:,0:num_labeled]
                I_minus_P_uu = np.identity(num_unlabeled) - P_uu_row_normalized

            try:
                f_u, f_l = inv_function(I_minus_P_uu, P_ul_row_normalized @ Y_l), Y_l
            except:
                print("Singular matrix")
                break
            

            function_diff = abs(f_u[x] - old_f_u[x])
            if function_diff < eps and sigma_diff < eps:
                # breaking, didn't find point
                break
            else:
                old_f_u = f_u
            
            g_u = (f_u - .5) ** 2
            if sparse:
                dw_dsigma = 2 * W_sigma.multiply(W_sparse.multiply(W_sparse)) / sigma ** 3
                dw_dsigma_row_sums = dw_dsigma.sum(axis=1)
                W_row_sums = W_sigma.sum(axis=1)
                dP_dsigma = sp.csr_matrix((dw_dsigma - P_sparse.multiply(dw_dsigma_row_sums)).multiply(1 / W_row_sums))                  
            else:
                dw_dsigma = 2 * W_sigma * W ** 2 / sigma ** 3
                dw_dsigma_row_sums = np.sum(dw_dsigma, axis=1, keepdims=True)
                W_row_sums = np.sum(W_sigma, axis=1, keepdims=True)
                dP_dsigma = (dw_dsigma - P * dw_dsigma_row_sums) / W_row_sums

            try:
                df_dsigma = inv_function(I_minus_P_uu, dP_dsigma[num_labeled:,num_labeled:] @ f_u + dP_dsigma[num_labeled:,0:num_labeled] @ f_l)
            except:
                print(f"Singular matrix")
                break
            dgu_dsigma = 2 * (f_u - .5) * df_dsigma

            # step closer to our solution
            Nesterov  = delta * dgu_dsigma[x]
            Newton = float("inf")
            if dgu_dsigma[x] != 0: Newton = 2 * g_u[x] / dgu_dsigma[x]


            if abs(Nesterov) < abs(Newton):
                new_sigma = sigma - Nesterov
                sigma_diff = abs(sigma - new_sigma)
                old_sigma = sigma

                lmda_s_1 = (1 + (1 + 4 * lmda_s ** 2) ** .5) / (2)
                gamma = (1 - lmda_s) / lmda_s_1
                sigma = (1 - gamma) * new_sigma + gamma * sigma
                lmda_s = lmda_s_1
            else:
                new_sigma = sigma - Newton
                sigma_diff = abs(sigma - new_sigma)
                old_sigma = sigma

                sigma = new_sigma
        overallcount += count

        # update interval if necessary
        if g_u[x] <  eps:
            if sigma > sigma_0:
                max_sigma = min(max_sigma, sigma)
            else:
                min_sigma = max(min_sigma, sigma)
    return min_sigma, max_sigma, overallcount

def find_all_approx_feedback_sets(sigma_0, sigma_end, W, Y, num_labeled, inv_function, interval_function, step_size=.05, epsilon=1e-4, delta=1, kNN=None, sparse=False):
    """given a starting and ending sigma, return intervals in that range

    Args:
        sigma_0 (int): start sigma value
        sigma_end (int): end sigma value
        W (ndarray): weight array before sigma
        Y (ndarray): label array
        num_labeled (int): number of labeled instances
        inv_function (function): function for inverse(i.e. full inverse or CG method)
        interval_function (function): function to use to calculate a single interval
        step_size (float, optional): step size for findind actual accuracy. Defaults to .1.
        epsilon (int, optional): paramater for interval finding algorithm. Defaults to 1e-3.
        delta (int, optional): parameter for interval finding algorithm. Defaults to 1.
        kNN (int, optional): parameter for determining kNN neighbor amount. If None, this is not used
        sparse (boolean, optional): whether to use sparse matricies when calculating intervals

    Returns:
        list, int: interval list with tuples for intervals, time to conduct interval algorithm
    """
    intervals = []
    sigma = sigma_0
    # create the intervals
    overallcount = 0
    num_intervals =  0
    overallTime = 0
    # find all intervals
    while sigma < sigma_end:
        time1 = time.time()
        min, maxVal, count  = interval_function(W, Y[0:num_labeled], sigma, inv_function, epsilon, delta, kNN=kNN, sparse=sparse)
        time2 = time.time()
        num_intervals += 1
        overallTime += (time2 - time1)
        overallcount += count
        intervals.append((min, maxVal))
        sigma = maxVal + step_size
    # bound intervals to (sigma_0, sigma_end)
    for x in range(len(intervals)):
        left, right = intervals[x]
        if left < sigma_0: intervals[x] = (sigma_0, intervals[x][1])
        if right > sigma_end: intervals[x] = (intervals[x][0], sigma_end)
    intervals = [intervals[0]] + [(max(intervals[index][0], intervals[index - 1][1]), intervals[index][1]) for index in range(1, len(intervals))]
    return intervals, overallTime, overallcount

def find_accuracy_values(sigma_0, sigma_end, W, Y, num_labeled, inv_function, problem_instance_function, accuracy_function, step_size=.05, kNN=None):
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
    arr = np.zeros(val_range.shape)

    for index, sigma in enumerate(val_range):
        sigma_matrix_array_inverse = np.exp((-W ** 2 / sigma**2))
        if kNN is not None:
            sigma_matrix_array_inverse = util.k_highest_neighbors(sigma_matrix_array_inverse, kNN)

        solution, time = problem_instance_function(sigma_matrix_array_inverse, Y[0:num_labeled], inv_function)
        accuracy, incorrect = accuracy_function(solution, Y[num_labeled:])
        # print(f"sigma: {sigma}, accuracy: {accuracy}, incorrect: {incorrect}")
        arr[index] = accuracy
    return arr

def plot_all_approx_feedback_sets_with_accuracy(sigma_0, sigma_end, intervals, ground_truth_accuracys, step_size=.05, images_path=None, intervals_path=None, time=None, count=None):
    """plots intervals overlayed with accuracys from a start to end sigmma range

    Args:
        sigma_0 (int): start sigma value
        sigma_end (int): end sigma value
        intervals (list): list of tuples containing intervals
        ground_truth_accuracys (ndarray): accuracys for each sigma values
        step_size (float, optional): step size that accuracys are said to be taken at. Defaults to .05.
        images_path (str, optional): path to save image result to. Defaults to None.
        intervals_path (str, optional): path to save inervals result to as a dictionary. Defaults to None.
        time (int, optional): amount of time spent on interval calculation for the interval dictionary
        count (int, optional): number of calls to an inverse/approximate inverse function while calculating all intervals for the interval dictionary
    """
    val_range = np.arange(sigma_0, sigma_end, step_size)

    assert val_range.shape == ground_truth_accuracys.shape, "Ground truth accuracys should have same shape as arange(sigma_0, sigma_end, step_size), where step size defaults to .05"

    errorVals, errorsizes = [((left + right) / 2) for left, right in intervals], [(right - left)/2 if right != left else .0001 for left, right in intervals]
    y_vals = [.005 + ground_truth_accuracys[np.argmin(np.array([abs(errorval - val) for val in val_range]))] for errorval in errorVals]
    y_vals = [y_val if y_val != float("inf") else sigma_end for y_val in y_vals]

    biglist = [util.errorbar(errorVals[x], y_vals[x], errorsizes[x], .005) for x in range(len(errorsizes))]
    lines = [elem for l in biglist for elem in l]
    
    
    lc = mc.LineCollection(lines, colors= "black")
    fig, ax = plt.subplots()
    ax.add_collection(lc)
    plt.plot(val_range, ground_truth_accuracys)
    plt.xlabel("sigma")
    plt.ylabel("accuracy")
    plt.savefig(images_path)
    plt.close()

    intervals_dict = {"ground_truth": list(ground_truth_accuracys), "sigma_vals": list(val_range), "intervals": intervals, "time": time, "count": count}
    with open(intervals_path, "w") as file:
        file.write(json.dumps(intervals_dict))

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        default="MNIST",
        help="dataset to use for generation. Default is MNIST. Other options are FashionMNIST and USPS"
    )
    parser.add_argument(
        "--num_experiments",
        type=int,
        default=1,
        help="number of independent experiments to run. Defaut is 1"
    )
    parser.add_argument(
        '--PCA', 
        action='store_true',
        default=False,
        help="whether or not to use PCA for cleaning data"
    )
    parser.add_argument(
        '--n_components', 
        type=int,
        default=None,
        help="number of principle components to use when doing PCA"
    )
    parser.add_argument(
        '--seed', 
        type=int,
        default=1,
        help="seed for generating subsets. Default is 1"
    )
    parser.add_argument(
        "--subset_size",
        type=int,
        default=100,
        help="subset size to use for generation. Default is 100"
    )
    parser.add_argument(
        "--num_labels",
        type=int,
        default=10,
        help="Number of labels for training. Default is 10"
    )
    parser.add_argument(
        "--sigma_start",
        type=float,
        default=1,
        help="sigma value to start finding intervals at. Default is 1"
    )
    parser.add_argument(
        "--sigma_end",
        type=float,
        default=7,
        help="sigma value to finish finding intervals at. Default is 7"
    )
    parser.add_argument(
        "--step_size",
        type=float,
        default=.05,
        help="sigma step size to use for calculating ground truth accuracys. Default is .05s"
    )
    parser.add_argument(
        "--inverse_type",
        type=str,
        default="CG",
        help="inverse type, should be either CG or full_inv"
    )
    parser.add_argument(
        "--CG_steps",
        type=int,
        default=None,
        help="number of steps in CG method. Default is 15"
    )
    parser.add_argument(
        "--kNN",
        type=int,
        default=None,
        help="k value to use kNN in accuracy and interval calculation. default is to not use kNN"
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-4,
        help="epsilon value to be used in algorithm as a stopping condition. Default is 1e-4"
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=1,
        help="step value to be used in algorithm for nesterov's accelerated GD. Default is 1"
    )
    parser.add_argument(
        "--images_path",
        type=str,
        default="images/",
        help="path to save images to. Default is images/"
    )
    parser.add_argument(
        "--intervals_path",
        type=str,
        default="intervals/",
        help="path to save intervals as .json files to. Default is intervals/"
    )
    args = parser.parse_args()

    if not args.PCA:
        assert args.n_components is None, \
            "set a number of components for PCA but not using PCA"

    assert args.inverse_type in ["CG", "full_inv"], "inverse type must be either CG or full_inv"
    
    if args.inverse_type == "CG":
        assert args.CG_steps is not None, "need to set number of steps in CG method with --CG_steps"
        args.inv = util.cg_inv(args.CG_steps)
        args.inv_string = f"CG_{args.CG_steps}"
        args.inv_string_title = f"CG {args.CG_steps}"
        if args.kNN is not None:
            args.sparse = True
            print("kNN graph with CG method toggled on - doing all interval calculation with sparse matricies")
        else:
            args.sparse = False
    else:
        args.inv = util.basic_inv
        args.inv_string = f"full_inv"
        args.inv_string_title = f"full inv"
        args.sparse = False
    
    args.kNN_string, args.kNN_path, args.PCA_path = "", "", ""
    if args.kNN is not None:
        args.kNN_string=f"kNN: {args.kNN}"
        args.kNN_path=f"_kNN_{args.kNN}"
    
    if args.PCA:
        if args.n_components is None:
            args.PCA_path = f"_PCA_86"
        else:
            args.PCA_path = f"_PCA_{args.n_components}"

    # can change labels subset to any set of two class labels for testing
    args.labels_subset = [0,1]
    args.interval_function = approx_feedback_set
    args.problem_instance_function = harmonic_approx
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
        labels_subset= args.labels_subset,
        num_subsets=args.num_experiments, 
        PCA=args.PCA, 
        PCA_n_components=args.n_components,
        seed=args.seed
    )
    for subset_index in range(W.shape[0]):
        subset_W, subset_Y = W[subset_index], Y[subset_index]
        print(f"Calculating intervals, subset {subset_index + 1} of {W.shape[0]}. dataset: {args.dataset}, seed {args.seed} ")
        intervals, time, count = find_all_approx_feedback_sets(
            sigma_0 = args.sigma_start,
            sigma_end=args.sigma_end, 
            W=subset_W,
            Y=subset_Y, 
            num_labeled=args.num_labels, 
            inv_function=args.inv, 
            interval_function=args.interval_function,
            step_size=args.step_size, 
            epsilon=args.epsilon, 
            delta=args.delta, 
            kNN=args.kNN,
            sparse=args.sparse,
        )
        ground_truth_accuracys = find_accuracy_values(
            sigma_0=args.sigma_start,
            sigma_end=args.sigma_end,
            W=subset_W,
            Y=subset_Y,
            num_labeled=args.num_labels,
            inv_function=args.inv,
            problem_instance_function=args.problem_instance_function,
            accuracy_function=args.accuracy_function, 
            step_size=args.step_size,
            kNN=args.kNN
        )

        plot_all_approx_feedback_sets_with_accuracy(
            sigma_0=args.sigma_start,
            sigma_end=args.sigma_end,
            intervals=intervals,
            ground_truth_accuracys=ground_truth_accuracys,
            step_size=args.step_size,
            images_path=f"{args.images_path}{args.dataset}_{args.subset_size}_{args.num_labels}_labels_seed_{args.seed + subset_index}_{args.inv_string}{args.kNN_path}{args.PCA_path}.png", 
            intervals_path=f"{args.intervals_path}{args.dataset}_{args.subset_size}_{args.num_labels}_labels_seed_{args.seed + subset_index}_{args.inv_string}{args.kNN_path}{args.PCA_path}.json",
            time=time,
            count=count, 
        )

if __name__ == "__main__":
    main()