# Note that in polyscope, y is upright, z is toward screen, x is to the right

import numpy as np
from src.visualize import polyscope_plot3d
from src.io import read_mesh, read_point_cloud, point_cloud_to_mesh
from src.ot_interpolation import sinkhorn, sinkhorn_log, ot_shape_interpolate, ot_multiple_shapes_interpolate
from src.utils import rotate_x, rotate_y, rotate_z, cube_voxelize, colorize, cube_to_point_clouds
from src.metrics import chamfer_distance, hausdorff_distance
from src.cost_matrix import calculate_kernel


def test_two_shapes_without_voxelization_tosca():
    # Interpolate and visualize
    point_cloud_1, triangles_1 = read_mesh("./data/meshes/tosca/david0.off")
    point_cloud_2, triangles_2 = read_mesh("./data/meshes/tosca/centaur0.off")
    shape_lst = ot_shape_interpolate(point_cloud_1, point_cloud_2, steps=8)
    for i in range(len(shape_lst)):
        shape = shape_lst[i][:, [0, 2, 1]].copy()
        shape[:, 2] = -shape[:, 2]
        shape_lst[i] = shape
    color_lst = []
    for i in range(len(shape_lst)):
        color_lst.append(np.linalg.norm(shape_lst[i], axis=1) / 100)
        shape_lst[i][:, 0] += 80 * i
        shape_lst[i][:, 2] -= 50 * i
        shape_lst[i] = rotate_y(shape_lst[i], -17)
        shape_lst[i] = rotate_x(shape_lst[i], -15)
    result_shape = np.vstack(shape_lst)
    result_color = np.hstack(color_lst)
    polyscope_plot3d(result_shape, result_color, 0.001)

    # Chamfer distance
    for i in range(len(shape_lst) - 1):
        print(chamfer_distance(shape_lst[i], shape_lst[i + 1]))
    print(chamfer_distance(shape_lst[0], shape_lst[-1]))

    # Hausdorff distance
    for i in range(len(shape_lst) - 1):
        print(hausdorff_distance(shape_lst[i], shape_lst[i + 1]))
    print(hausdorff_distance(shape_lst[0], shape_lst[-1]))


def test_two_shapes_with_voxelization_tosca():
    # Interpolate
    point_cloud_1, triangles_1 = read_mesh("./data/meshes/tosca/cat0.off")
    point_cloud_2, triangles_2 = read_mesh("./data/meshes/tosca/dog0.off")
    shape_lst = ot_shape_interpolate(point_cloud_1, point_cloud_2, steps=8)
    for i in range(len(shape_lst)):
        shape = shape_lst[i].copy()
        shape, _, _ = cube_voxelize(shape, 500, (-4.1, 15.2, 142.2), 385)
        shape = cube_to_point_clouds(shape)
        shape = shape[:, [0, 2, 1]]
        shape[:, 2] = -shape[:, 2]
        shape_lst[i] = shape

    # Chamfer distance
    for i in range(len(shape_lst) - 1):
        print(chamfer_distance(shape_lst[i], shape_lst[i + 1]))
    print(chamfer_distance(shape_lst[0], shape_lst[-1]))

    # Hausdorff distance
    for i in range(len(shape_lst) - 1):
        print(hausdorff_distance(shape_lst[i], shape_lst[i + 1]))
    print(hausdorff_distance(shape_lst[0], shape_lst[-1]))

    # Visualize
    color_lst = []
    for i in range(len(shape_lst)):
        color_lst.append(colorize(shape_lst[i]))
        shape_lst[i][:, 0] += 80 * i
        shape_lst[i][:, 2] -= 50 * i
        shape_lst[i] = rotate_y(shape_lst[i], -17)
        shape_lst[i] = rotate_x(shape_lst[i], -15)
    result_shape = np.vstack(shape_lst)
    result_color = np.hstack(color_lst)
    polyscope_plot3d(result_shape, result_color, 0.001)


def test_voxelization_tosca():
    pd, _ = read_mesh("./data/meshes/tosca/horse0.off")

    shape, _, _ = cube_voxelize(pd, 1000, (-4.1, 15.2, 142.2), 385)
    shape = cube_to_point_clouds(shape)
    color = colorize(shape)

    polyscope_plot3d(shape, color, quad_radius=0.001)


def test_barycenter_two_shapes_tosca():
    point_cloud_1, triangles_1 = read_mesh("./data/meshes/tosca/david0.off")
    point_cloud_2, triangles_2 = read_mesh("./data/meshes/tosca/centaur0.off")

    shape_1, _, _ = cube_voxelize(point_cloud_1, 100, (-4.1, 15.2, 142.2), 385)
    shape_2, _, _ = cube_voxelize(point_cloud_2, 100, (-4.1, 15.2, 142.2), 385)

    # shape = shape_1.copy()
    # # for i in range(shape.shape[0]):
    # #     shape[0, :, :] = calculate_kernel(shape[0, :, :])
    # shape = calculate_kernel(shape)
    # shape = cube_to_point_clouds(shape)
    # polyscope_plot3d(shape, colorize(shape), 0.001)

    scaling_factor = 1

    shape_1 *= scaling_factor
    shape_2 *= scaling_factor

    barycenter_lst, lambda_lst_lst = ot_multiple_shapes_interpolate([shape_1 + 0.01, shape_2 + 0.01], 3)

    for i in range(len(barycenter_lst)):
        barycenter = barycenter_lst[i].copy() / scaling_factor
        barycenter = cube_to_point_clouds(barycenter)
        # barycenter = barycenter[:, [0, 2, 1]]
        barycenter[:, 2] = -barycenter[:, 2]
        barycenter_lst[i] = barycenter

    # Visualize
    color_lst = []
    for i in range(len(barycenter_lst)):
        color_lst.append(colorize(barycenter_lst[i]))
        # barycenter_lst[i][:, 0] += 80 * i
        # barycenter_lst[i][:, 2] -= 50 * i
        barycenter_lst[i][:, 2] -= lambda_lst_lst[i][0] * 200
        barycenter_lst[i][:, 0] -= lambda_lst_lst[i][1] * 200
        # barycenter_lst[i][:, 0] += lambda_lst_lst[i][2] * 200

        # shape_lst[i] = rotate_y(shape_lst[i], -17)
        # shape_lst[i] = rotate_x(shape_lst[i], -15)
    result_shape = np.vstack(barycenter_lst)
    result_color = np.hstack(color_lst)
    polyscope_plot3d(result_shape, result_color, 0.001)


def test_barycenter_three_shapes_tosca():
    point_cloud_1, triangles_1 = read_mesh("./data/meshes/tosca/david0.off")
    point_cloud_2, triangles_2 = read_mesh("./data/meshes/tosca/centaur0.off")
    point_cloud_3, triangles_3 = read_mesh("./data/meshes/tosca/horse0.off")

    shape_1, _, _ = cube_voxelize(point_cloud_1, 100, (-4.1, 15.2, 142.2), 385)
    shape_2, _, _ = cube_voxelize(point_cloud_2, 100, (-4.1, 15.2, 142.2), 385)
    shape_3, _, _ = cube_voxelize(point_cloud_3, 100, (-4.1, 15.2, 142.2), 385)

    barycenter_lst, lambda_lst_lst = ot_multiple_shapes_interpolate([shape_1 + 0.01, shape_2 + 0.01, shape_3 + 0.01], 5)

    for i in range(len(barycenter_lst)):
        barycenter = barycenter_lst[i].copy()
        barycenter = cube_to_point_clouds(barycenter)
        # barycenter = barycenter[:, [0, 2, 1]]
        barycenter[:, 2] = -barycenter[:, 2]
        barycenter_lst[i] = barycenter

    # Visualize
    color_lst = []
    for i in range(len(barycenter_lst)):
        color_lst.append(colorize(barycenter_lst[i]))
        # barycenter_lst[i][:, 0] += 80 * i
        # barycenter_lst[i][:, 2] -= 50 * i
        barycenter_lst[i][:, 2] -= lambda_lst_lst[i][0] * 200
        barycenter_lst[i][:, 0] -= lambda_lst_lst[i][1] * 200
        barycenter_lst[i][:, 0] += lambda_lst_lst[i][2] * 200

        # shape_lst[i] = rotate_y(shape_lst[i], -17)
        # shape_lst[i] = rotate_x(shape_lst[i], -15)
    result_shape = np.vstack(barycenter_lst)
    result_color = np.hstack(color_lst)
    polyscope_plot3d(result_shape, result_color, 0.001)


def test_barycenter_two_images():
    import matplotlib.pyplot as plt
    img_1 = plt.imread("disk.bmp")
    img_2 = plt.imread("letter-z.bmp")

    barycenter_lst, lambda_lst_lst = ot_multiple_shapes_interpolate([img_1 + 0.01, img_2 + 0.01], 10)
    print(lambda_lst_lst)
    print(np.max(img_1))
    print(np.min(img_1))

    for barycenter in barycenter_lst:
        plt.imshow(barycenter)
        plt.show()