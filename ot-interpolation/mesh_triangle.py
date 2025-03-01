import open3d as o3d
import numpy as np
from generate_data import generate_point_cloud

def conformal_distortion(A, B):
    print(A.shape)
    print(B.shape)

    point_cloud_1 = o3d.geometry.PointCloud()
    point_cloud_2 = o3d.geometry.PointCloud()

    # Assign points to the PointCloud object
    point_cloud_1.points = o3d.utility.Vector3dVector(A)
    point_cloud_2.points = o3d.utility.Vector3dVector(B)

    # Estimate normals
    point_cloud_1.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    point_cloud_2.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    # Compute the triangle mesh using Poisson surface reconstruction
    mesh_1, densities_1 = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(point_cloud_1, depth=3)
    mesh_2, densities_2 = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(point_cloud_2, depth=3)

    # Remove low-density vertices to get a cleaner mesh
    # vertices_to_remove_1 = densities_1 < np.quantile(densities_1, 0.01)
    # mesh_1.remove_vertices_by_mask(vertices_to_remove_1)
    # vertices_to_remove_2 = densities_2 < np.quantile(densities_2, 0.01)
    # mesh_2.remove_vertices_by_mask(vertices_to_remove_2)

    # Extract the list of triangles
    triangles_1 = np.asarray(mesh_1.triangles)
    triangles_2 = np.asarray(mesh_2.triangles)

    value_1 = 0
    for triangle_idx in triangles_1:
        # print(triangle_idx)
        triangle = np.array([A[triangle_idx[0]], A[triangle_idx[1]], A[triangle_idx[2]]])
        value_1 += (1/6) * (- triangle[2, 0] * triangle[1, 1] * triangle[0, 2]
                            + triangle[1, 0] * triangle[2, 1] * triangle[0, 2]
                            + triangle[2, 0] * triangle[0, 1] * triangle[1, 2]
                            - triangle[0, 0] * triangle[2, 1] * triangle[1, 2]
                            - triangle[1, 0] * triangle[0, 1] * triangle[2, 2]
                            + triangle[0, 0] * triangle[1, 1] * triangle[2, 2])


    value_2 = 0
    for triangle_idx_2 in triangles_2:
        # print(triangle_idx_2)
        triangle_2 = np.array([B[triangle_idx_2[0]], B[triangle_idx_2[1]], B[triangle_idx_2[2]]])
        value_2 += (1/6) * (- triangle_2[2, 0] * triangle_2[1, 1] * triangle_2[0, 2]
                            + triangle_2[1, 0] * triangle_2[2, 1] * triangle_2[0, 2]
                            + triangle_2[2, 0] * triangle_2[0, 1] * triangle_2[1, 2]
                            - triangle_2[0, 0] * triangle_2[2, 1] * triangle_2[1, 2]
                            - triangle_2[1, 0] * triangle_2[0, 1] * triangle_2[2, 2]
                            + triangle_2[0, 0] * triangle_2[1, 1] * triangle_2[2, 2])

    return abs(value_1 - value_2)


