import numpy as np
import os
import scipy.sparse.linalg as lin
from sklearn.decomposition import PCA
import math

import torchvision
from torchvision import transforms
from torchvision.transforms import ToTensor, Normalize

# in order to download USPS dataset
import ssl

ssl._create_default_https_context = ssl._create_unverified_context


def inplace_csr_row_normalize_l1(X):
    """Inplace row normalize using the l1 norm"""
    _inplace_csr_row_normalize_l1(X.data, X.shape, X.indices, X.indptr)


def _inplace_csr_row_normalize_l1(X_data, shape, X_indices, X_indptr):
    n_samples = shape[0]

    # the column indices for row i are stored in:
    #    indices[indptr[i]:indices[i+1]]
    # and their corresponding values are stored in:
    #    data[indptr[i]:indptr[i+1]]

    for i in range(n_samples):
        sum_ = 0.0

        for j in range(X_indptr[i], X_indptr[i + 1]):
            sum_ += abs(X_data[j])

        if sum_ == 0.0:
            # do not normalize empty rows (can happen if CSR is not pruned
            # correctly)
            continue

        for j in range(X_indptr[i], X_indptr[i + 1]):
            X_data[j] /= sum_


def errorbar(midpointX, midpointY, distance, height):
    """creates three segments corresponding to an error bar at some midpoint coordinate.
    Size of error bar paramaterized by distance and height"""
    return [
        [(midpointX - distance, midpointY), (midpointX + distance, midpointY)],
        [
            (midpointX - distance, midpointY + height),
            (midpointX - distance, midpointY - height),
        ],
        [
            (midpointX + distance, midpointY + height),
            (midpointX + distance, midpointY - height),
        ],
    ]


def segment(point, slope, size):
    """creates a derivative segment at a given point with a certain slope and size"""
    x, y = point
    theta = math.atan(slope)
    x_prime = size * math.cos(theta)
    y_prime = size * math.sin(theta)
    return [(x - x_prime / 2, y - y_prime / 2), (x + x_prime / 2, y + y_prime / 2)]


def l2_dist(x, y):
    return np.linalg.norm(x - y)


def basic_inv(A, b):
    return np.linalg.inv(A) @ b


def basic_inv_wrapped(x):
    return basic_inv


def cg_inv(x=20):
    """returns function that does Conjugate Gradient method with x iterations

    Args:
        x (int, optional): number of iterations for CG method on function return. Defaults to 20.
    """

    def newfunc(A, b):
        sol, _ = lin.cg(A, b, maxiter=x)
        return sol

    return newfunc


def measure_accuracy(guess_arr, actual_arr, threshold=0.5):
    """returns accuracy of guess_arr with ground truth actual_arr,
       assuming binary classification of < threshold and > threshold

    Args:
        guess_arr (ndarray)
        actual_arr (ndarray)
        threshold (ndarray)

    Returns:
        _type_: accuracy, array of incorrect examples
    """
    answers_array = np.logical_not(np.logical_xor(guess_arr > threshold, actual_arr > threshold))
    (incorrect,) = np.where(np.logical_not(answers_array))
    return np.sum(answers_array) / guess_arr.shape[0], incorrect


def truncate_bottom_k(W, k):
    """returns a mask of all but lowest k values per row of W

    Args:
        W (matrix): graph matrix
        k (int): value for kNN

    Returns:
        boolean ndarray: False for only bottom k values per row
    """
    m, n = W.shape
    # get (unsorted) indices of top-k values
    topk_indices = np.argpartition(W, k, axis=1)[:, 0:k]
    # get k-th value
    rows, _ = np.indices((m, k))
    kth_vals = W[rows, topk_indices].max(axis=1)
    # get boolean mask of values smaller than k-th
    bad_vals = W > kth_vals[:, None]
    # replace mask by 0
    return bad_vals


def truncate_top_k(W, k):
    """returns a mask of all but highest k values per row of W

    Args:
        W (matrix): graph matrix
        k (int): value for kNN

    Returns:
        boolean ndarray: False for only top k values per row
    """
    m, n = W.shape
    # get (unsorted) indices of top-k values
    topk_indices = np.argpartition(W, W.shape[0] - k, axis=1)[:, -k:]
    # get k-th value
    rows, _ = np.indices((m, k))
    kth_vals = W[rows, topk_indices].min(axis=1)
    # get boolean mask of values smaller than k-th
    bad_vals = W < kth_vals[:, None]
    # replace mask by 0
    return bad_vals


def k_lowest_neighbors(W, k):
    """given matrix W, zeros out any entry i,j that is not in the k lowest neighbors for either node i or j

    Args:
        k (int):
        W (ndarray):

    Returns:
        ndarray: W with correct values masked out
    """
    row_top_k_mask = truncate_bottom_k(W, k)
    col_top_k_mask = truncate_bottom_k(W.T, k)
    symmetric_mask = np.minimum(row_top_k_mask, col_top_k_mask.T)
    W[symmetric_mask] = 0
    return W


def k_highest_neighbors(W, k):
    """given matrix W, zeros out any entry i,j that is not in the k highest neighbors for either node i or j

    Args:
        k (int):
        W (ndarray):

    Returns:
        ndarray: W with correct values masked out
    """
    row_top_k_mask = truncate_top_k(W, k)
    col_top_k_mask = truncate_top_k(W.T, k)
    symmetric_mask = np.minimum(row_top_k_mask, col_top_k_mask.T)
    W[symmetric_mask] = 0
    return W


def load_pytorch_dataset(name):
    """load pytorch dataset given name

    Args:
        name (string): dataset to load

    Returns:
        ndarray, ndarray: training data flattened to num_elements*overall_length, labels
    """
    assert name in ["MNIST", "FashionMNIST", "Omniglot", "USPS", "CIFAR"]
    transform = torchvision.transforms.ToTensor()

    if name == "MNIST":
        trainset = torchvision.datasets.MNIST(
            root="/datasets/", download=True, train=True, transform=transform
        )
    elif name == "FashionMNIST":
        trainset = torchvision.datasets.FashionMNIST(
            root="/datasets/", download=True, train=True, transform=transform
        )
    elif name == "CIFAR":
        transform = transforms.Compose(
            [ToTensor(), Normalize([0.4914, 0.4822, 0.4465], [0.2471, 0.2435, 0.2616])]
        )
        trainset = torchvision.datasets.CIFAR10(
            root="/datasets/", download=True, train=False, transform=transform
        )
    elif name == "USPS":
        trainset = torchvision.datasets.USPS(
            root="/datasets/", download=True, train=True, transform=transform
        )
    else:
        trainset = torchvision.datasets.Omniglot(
            root="/datasets/", download=True, transform=transform
        )

    if name == "CIFAR":
        X_train = np.zeros((len(trainset), 3, 32, 32))
    else:
        X_train = np.zeros((len(trainset), trainset[0][0].shape[1], trainset[0][0].shape[2]))
    y_train = np.zeros((len(trainset)))

    if name == "Omniglot":
        y_train = [
            trainset._characters[character_class].split("\\")[0]
            for image_name, character_class in trainset._flat_character_images
        ]
        curr_number = 0
        curr_value = y_train[0]
        for index in range(len(y_train)):
            if y_train[index] != curr_value:
                curr_number += 1
                curr_value = y_train[index]
            y_train[index] = curr_number
        y_train = np.array(y_train)

    for index in range(len(trainset)):
        data, label = trainset[index]
        X_train[index] = data
        if name != "Omniglot":
            y_train[index] = label

    if name != "CIFAR":
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1] * X_train.shape[2]))
    return X_train, y_train


def create_matrix_from_training_data(X, distance_func):
    """create graph matrix based on distance function

    Args:
        X (ndarray): data vectors
        distance_func (function): function to do distance with

    Returns:
        ndarray: weight matrix Wwhere W_{i,j} is the distance between points i and j
    """
    matrix = np.zeros((X.shape[0], X.shape[0]))
    for row in range(X.shape[0]):
        for col in range(X.shape[0]):
            matrix[row][col] = distance_func(X[row], X[col])
    return matrix


def transform_labels_to_binary(labels, subset):
    """given a set of labels, make the values in subset equal to 1 and the rest 0"""
    for index in range(len(labels)):
        if labels[index] in subset:
            labels[index] = 1
        else:
            labels[index] = 0
    return labels


def create_certain_labels_mask(labels, subset):
    """given a set of labels, return a mask for those labels in subset"""
    mask = np.zeros(labels.shape[0]).astype(bool)
    for index in range(len(labels)):
        if labels[index] in subset:
            mask[index] = True
    return mask


def perform_PCA(X, n_components, path_tag, save=True):
    """given some datapoints and a number of components, generate the PCA for this data, and
       save it in {path_tag}_PCA.npy if save is set to true. If there is already something in
       this location, just load that data

    Args:
        X (ndarray): dataset
        n_components (int): number of PCA components to save
        path_tag (string): path_tag when savving the PCA components
        save (bool, optional): whether or not to save the PCA. Defaults to True.

    Returns:
        _type_: _description_
    """
    if not os.path.exists(f"PCA_files/{path_tag}_PCA.npy"):
        print("creating principle components to be saved in PCA_file folder...")
        ncomponents = n_components
        pca = PCA(n_components=ncomponents)
        principle_components = pca.fit_transform(X)
        if not os.path.exists("PCA_files"):
            os.mkdir("PCA_files")
        if save:
            with open(f"PCA_files/{path_tag}_PCA.npy", "wb") as f:
                np.save(f, principle_components)
    else:
        print(f"loading principle components from {path_tag}_PCA.npy")
        with open(f"PCA_files/{path_tag}_PCA.npy", "rb") as f:
            principle_components = np.load(f)

    return principle_components


def setup_delalleau_experiment(
    dataset,
    subset_size,
    extension_size,
    labels_subset=None,
    num_subsets=1,
    PCA=False,
    PCA_n_components=None,
    seed=0,
):
    if PCA and PCA_n_components is None:
        print("Warning: no number of PCA components given - using default 86 dimensions")
        PCA_n_components = 86
    X, Y = load_pytorch_dataset(dataset)

    if PCA:
        X = perform_PCA(X, PCA_n_components, dataset)

    if labels_subset is not None:
        labels_mask = create_certain_labels_mask(Y, labels_subset)
        X, Y = X[labels_mask], Y[labels_mask]

    n, datapoint_size = X.shape
    results_X, results_Y = np.zeros((num_subsets, subset_size, subset_size)), np.zeros(
        (num_subsets, subset_size + extension_size)
    )
    values_main, values_extended = np.zeros((num_subsets, subset_size, datapoint_size)), np.zeros(
        (num_subsets, extension_size, datapoint_size)
    )

    for subset in range(num_subsets):
        np.random.seed(seed + subset)
        shuffle = np.arange(n)
        np.random.shuffle(shuffle)
        curr_subset_X, curr_subset_Y, curr_subset_extended = (
            np.copy(X[shuffle][0:subset_size]),
            np.copy(Y[shuffle][0 : subset_size + extension_size]),
            np.copy(X[shuffle][subset_size : subset_size + extension_size]),
        )
        results_X[subset] = create_matrix_from_training_data(curr_subset_X, l2_dist)
        results_Y[subset] = curr_subset_Y
        values_main[subset] = curr_subset_X
        values_extended[subset] = curr_subset_extended

    return results_X, results_Y, values_main, values_extended


def get_random_subsets(
    dataset,
    subset_size,
    labels_subset=None,
    num_subsets=1,
    PCA=False,
    PCA_n_components=None,
    seed=0,
):
    """return a list of random subsets from a given dataset

    Args:
        dataset (string): the dataset to retreive from (as of right now, MNIST, FashionMNIST, or CIFAR10)
        subset_size (int): subset size for our data
        labels_subset (list, optional): if not using all data, set of labels to consider (e.g [0,1]). Defaults to None.
        num_subsets (int, optional): number of independent subsets to generate. Defaults to 1.
        PCA (bool, optional): whether or not to use PCA for this dataset. Defaults to False.
        PCA_n_components (int, optional): number of PCA components to use. If none, set to 86 as in Large Graph Construction paper. Defaults to None.
        seed (int, optional): seed to create subsets. Defaults to 0.

    Returns:
        X,Y: ndarrays where X[i], Y[i] is the ith instance of the problem with X[i] being the weight matrix and Y[i] being the labels
    """
    if PCA and PCA_n_components is None:
        print("Warning: no number of PCA components given - using default 86 dimensions")
        PCA_n_components = 86
    X, Y = load_pytorch_dataset(dataset)

    if PCA:
        X = perform_PCA(X, PCA_n_components, dataset)

    if labels_subset is not None:
        labels_mask = create_certain_labels_mask(Y, labels_subset)
        X, Y = X[labels_mask], Y[labels_mask]

    n = X.shape[0]
    results_X, results_Y = np.zeros((num_subsets, subset_size, subset_size)), np.zeros(
        (num_subsets, subset_size)
    )

    for subset in range(num_subsets):
        np.random.seed(seed + subset)
        shuffle = np.arange(n)
        np.random.shuffle(shuffle)
        curr_subset_X, curr_subset_Y = np.copy(X[shuffle][0:subset_size]), np.copy(
            Y[shuffle][0:subset_size]
        )
        results_X[subset] = create_matrix_from_training_data(curr_subset_X, l2_dist)
        results_Y[subset] = curr_subset_Y

    return results_X, results_Y
