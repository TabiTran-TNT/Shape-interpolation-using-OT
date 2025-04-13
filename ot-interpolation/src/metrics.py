import numpy as np
from scipy.spatial import KDTree


def chamfer_distance(point_cloud_1, point_cloud_2):
    tree = KDTree(point_cloud_2)
    dist_1 = tree.query(point_cloud_1)[0]
    tree = KDTree(point_cloud_1)
    dist_2 = tree.query(point_cloud_2)[0]
    return 0.5 * np.mean(dist_1) + 0.5 * np.mean(dist_2)

def hausdorff_distance(point_cloud_1, point_cloud_2):
    tree = KDTree(point_cloud_2)
    dist_1 = tree.query(point_cloud_1)[0]
    tree = KDTree(point_cloud_1)
    dist_2 = tree.query(point_cloud_2)[0]
    return 0.5 * np.max(dist_1) + 0.5 * np.max(dist_2)