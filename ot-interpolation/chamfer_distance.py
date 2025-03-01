import numpy as np
from scipy.spatial import KDTree

def chamfer_distance(A, B):
    """
    Computes the chamfer distance between two sets of points A and B.
    """
    tree = KDTree(B)
    dist_A = tree.query(A)[0]
    tree = KDTree(A)
    dist_B = tree.query(B)[0]
    return 0.5 * np.mean(dist_A) + 0.5 * np.mean(dist_B)

def hausdorff_distance(A, B):
    """
    Computes the chamfer distance between two sets of points A and B.
    """
    tree = KDTree(B)
    dist_A = tree.query(A)[0]
    tree = KDTree(A)
    dist_B = tree.query(B)[0]
    return 0.5 * np.max(dist_A) + 0.5 * np.max(dist_B)1