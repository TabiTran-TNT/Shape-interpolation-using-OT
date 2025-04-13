import numpy as np


def rotate_x(points, deg=0):
    rad = (deg/180)*2*np.pi
    rotation_matrix = np.array([[1, 0, 0],
                                [0, np.cos(rad), -np.sin(rad)],
                                [0, np.sin(rad), np.cos(rad)]])
    return points @ rotation_matrix


def rotate_y(points, deg=0):
    rad = (deg/180)*2*np.pi
    rotation_matrix = np.array([[np.cos(rad), 0, -np.sin(rad)],
                                [0, 1, 0],
                                [np.sin(rad), 0, np.cos(rad)]])
    return points @ rotation_matrix


def rotate_z(points, deg=0):
    rad = (deg/180)*2*np.pi
    rotation_matrix = np.array([[np.cos(rad), np.sin(rad), 0],
                                [-np.sin(rad), np.cos(rad), 0],
                                [0, 0, 1]])
    return points @ rotation_matrix


def cube_voxelize(points:np.ndarray, n=100, center=(0, 0, 0), width=1):
    shape = np.zeros((n, n, n))
    gap = width / n
    for point in points:
        i = int((point[0] - center[0] + width / 2) // gap)
        if i < 0 or i >= n:
            continue
        j = int((point[1] - center[1] + width / 2) // gap)
        if j < 0 or j >= n:
            continue
        k = int((point[2] - center[2] + width / 2) // gap)
        if k < 0 or k >= n:
            continue
        shape[i, j, k] = 1.0
    return shape, center, width


def colorize(points:np.ndarray):
    color = np.zeros(points.shape[0])
    for i in range(points.shape[0]):
        color[i] = np.linalg.norm(points[i])
    return color


def cube_to_point_clouds(shape:np.ndarray):
    indexes = np.where(shape >= 0.05)
    point_cloud = np.array(list(zip(indexes[0], indexes[1], indexes[2]))).astype("float64")
    return point_cloud
