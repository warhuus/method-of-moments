from typing import List

import numpy as np
import scipy.linalg


def form_L(B312: List[np.ndarray], k: int, verbose: bool) -> np.ndarray:
    ''' Return L, R3 '''
    L = np.empty((k, k))

    # step 1: compute R3 that diagonalizes B312[0]
    L[0], R3 = scipy.linalg.eig(B312[0])
    R3 = R3.real
    
    # step 2: obtain the diagonals of the matrices R3^-1 @ B312[j] @ R3
    # for all but the first entry.
    try:
        R3_inv = np.linalg.inv(R3)
    except np.linalg.LinAlgError:
        return None, None
        if verbose:
            print(f'failed to invert R3:\n\nR3 = {R3}\n')

    for i in range(1, k):
        L[i] = np.diag(R3_inv.dot(B312[i]).dot(R3))
    
    return L, R3


def sample_rotation_matrix(k: int) -> np.ndarray:
    ''' Make sample rotation matrix Theta '''
    theta, R = np.linalg.qr(np.random.normal(size=(k, k)))
    theta = theta @ np.diag(np.sign(np.diag(R)))

    if np.linalg.det(theta) < 0:
        theta[:, [0, 1]] = theta[:, [1, 0]]

    return theta


def make_P32(X: np.ndarray) -> np.ndarray:
    ''' Compute P32 '''
    P32 = [np.einsum('i, j -> ij', X[i + 2], X[i + 1])
           for i in range(len(X) - 2)]
        
    return sum(P32) / len(P32)


def make_P31(X: np.ndarray) -> np.ndarray:
    ''' Compute P31 '''
    P31 = [np.einsum('i, j -> ij', X[i + 2], X[i])
           for i in range(len(X) - 2)]
    return sum(P31) / len(P31)


def make_P312(X: np.ndarray) -> np.ndarray:
    ''' Compute P_312 '''
    P312 = [np.einsum('i, j, k -> ijk', X[i + 2], X[i], X[i + 1])
            for i in range(0, len(X) - 2)]
    return sum(P312) / len(P312)


def run(X: np.ndarray, k: int, verbose: bool = False) -> np.ndarray:
    '''
    Implementation of Algorithm B from Anandkumar, et al. 2012 for HMMs with
    multivariate Gaussian emissions. Code taken partly from maxentile (https://bit.ly/3ualJru)
    with inspiration from cmgithub's Matlab code for discrete emissions (https://bit.ly/3uakvfW).
    
    Returns None, None upon insolvent results. To see errors, use verbose = True.

    Input:
        X: Time series data, ndarray of shape (sample size, dimesions).
        k: A prior number of states, integer.
        verbose: boolean (default False), whether or not to print errors.
        
    Output:
        O: Estimated emission means, ndarray of shape (k, dimensions).
        T: Transition probability matrix, ndarray of shape (k, k). The
            probability of transitioning from the i'th to the j'th state
            is given by cell (j, i).

    '''

    # make P matrices
    P31 = make_P31(X)
    P32 = make_P32(X)
    P312 = make_P312(X)

    # compute top-k singular vectors
    U3, _, U1 = np.linalg.svd(P31)
    _, _, U2 = np.linalg.svd(P32)

    U1 = U1[:, :k]
    U2 = U2[:, :k]
    U3 = U3[:, :k]

    # pick invertible theta
    theta = sample_rotation_matrix(k)

    # form B312(U2 theta_j) for all j
    P312_U3_theta = [P312.dot(U2.dot(theta[j])) for j in range(k)]
    B312 = [
        (U3.T.dot(P312_U3_theta[j]).dot(U1)).dot(
        np.linalg.inv(U3.T.dot(P31).dot(U1)))
        for j in range(k)
    ]

    # form matrix L
    L, R3 = form_L(B312, k, verbose)

    if L is None:
        return None, None

    else:

        # form and return M2
        O = U2.dot(np.linalg.inv(theta).dot(L))

        # get transition matrix
        try:
            T = np.linalg.inv(U3.T.dot(O)).dot(R3)
        except np.linalg.LinAlgError:
            T = None
            if verbose:
                print(f'failed to invert U3^T O:\n\nU3^T O = {U3.T.dot(O)}\n')

        # set to None if any probabilities are negative
        if (T < 0).any():
            T = None
            if verbose:
                print(f'negative probability in T:\n\nT = {T}\n')
        else:
            T = T / T.sum(axis=0).T

        return O, T
