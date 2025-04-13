import open3d as o3d
import numpy as np


def generate_point_cloud(name="duck", sparse_ratio=1):
    if name == "duck":
        return get_duck_point_cloud(sparse_ratio)
    elif name == "donut":
        return get_donut_point_cloud(sparse_ratio)
    else:
        return get_duck_point_cloud(sparse_ratio)


def get_duck_point_cloud(sparse_ratio):
    pcd = o3d.io.read_point_cloud("data/point-clouds/514_template.pcd")

    points = np.asarray(pcd.points)
    points = points[:, [1, 2, 0]]
    points[:, 2] = -points[:, 2]
    for i in range(3):
        points[:, i] -= (np.max(points[:, i]) - np.min(points[:, i])) / 2 + np.min(points[:, i])

    max_gap = -1e9
    for i in range(3):
        max_gap = max(np.max(points[:, i]) - np.min(points[:, i]), max_gap)

    for i in range(3):
        points[:, i] /= max_gap

    points = points[::sparse_ratio, :]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud("data/point-clouds/duck-2000.pcd", pcd)


def get_donut_point_cloud(sparse_ratio):
    big_r = 13
    small_r = 6
    n = 45  # Equal to square root of number of points

    # Generate points
    theta = np.linspace(0, 2 * np.pi, n)
    phi = np.linspace(0, 2 * np.pi, n)
    theta, phi = np.meshgrid(theta, phi)
    theta = theta.flatten()
    phi = phi.flatten()

    x = (big_r + small_r * np.cos(phi)) * np.cos(theta)
    y = (big_r + small_r * np.cos(phi)) * np.sin(theta)
    z = small_r * np.sin(phi)

    points = np.vstack((x, y, z)).T
    max_gap = -1e9
    for i in range(3):
        max_gap = max(np.max(points[:, i]) - np.min(points[:, i]), max_gap)

    for i in range(3):
        points[:, i] /= max_gap

    points = points[::sparse_ratio, :]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud("data/point-clouds/donut-2000.pcd", pcd)
