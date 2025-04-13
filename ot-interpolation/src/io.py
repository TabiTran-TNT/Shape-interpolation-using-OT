import numpy as np
import open3d as o3d


def read_mesh(path):
    mesh = o3d.io.read_triangle_mesh(path)
    points = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    return points, triangles


def read_point_cloud(path):
    point_cloud = o3d.io.read_point_cloud(path)
    points = np.asarray(point_cloud.points)
    return points


def point_cloud_to_mesh(point_cloud):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    # pcd.normals = o3d.utility.Vector3dVector(np.zeros((1, 3)))
    pcd.estimate_normals()
    pcd.orient_normals_consistent_tangent_plane(20)
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd=pcd)
    vertices_to_remove = densities < np.quantile(densities, 0.005)
    mesh.remove_vertices_by_mask(vertices_to_remove)
    return np.asarray(mesh.vertices), np.asarray(mesh.triangles)

