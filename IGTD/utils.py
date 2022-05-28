import numpy as np
from scipy.stats import entropy

def normalized_data(data):
    """
        return normlized data where each column represent distribution range [0,1]
    """
    return data / data.sum(axis=0, keepdims=1)

def jsd_matrix(X: np.ndarray)-> np.ndarray:
    """
        Calculate the Jensen Shannon Distance for all pairs of features in X.
        :param X: data - size of MxN 
        :return: The similarity matrix - size of NxN
    """
    n = X.shape[1]
    similarity_mat = np.zeros((n,n))
    for i in range(n):
        for j in range(i+1):
            similarity_mat[i,j] = jensen_shannon_distance(X[:, i], X[:, j])
            similarity_mat[j,i] = similarity_mat[i,j]
    return similarity_mat

def jensen_shannon_distance(x1: np.ndarray, x2: np.ndarray) -> float:
    """
        JSD measure the similirity of X1's distribution and X2's distribution.
        The Jensen Shannon distance is the square root of Jensen Shannon divergence.
        JSD(X1||X2) = sqrt( 0.5 * KL(X1||M) + 0.5 * D(X2||M) where M = 0.5 * (X1 + X2) )
        Range of JSD : [0,1]
        :param x1: feature of X
        :param x2: feature of X
        :return mesure of similirity between X1's distribution and X2's distribution.
    """
    m = 0.5 * (x1 + x2)
    kl_div_x1m = entropy(x1, m)
    kl_div_x2m = entropy(x2, m)
    return np.sqrt(0.5*kl_div_x1m + 0.5*kl_div_x2m)
