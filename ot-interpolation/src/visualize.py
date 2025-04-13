import numpy as np
import matplotlib.pyplot as plt
import polyscope as ps


def matplotlib_plot3d(points, axis_on=False, look_down="default"):
    def set_axes_equal(_ax):
        limits = np.array([
            _ax.get_xlim3d(),
            _ax.get_ylim3d(),
            _ax.get_zlim3d(),
        ])
        origin = np.mean(limits, axis=1)
        radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
        _set_axes_radius(_ax, origin, radius)

    def _set_axes_radius(_ax, origin, radius):
        x, y, z = origin
        _ax.set_xlim3d([x - radius, x + radius])
        _ax.set_ylim3d([y - radius, y + radius])
        _ax.set_zlim3d([z - radius, z + radius])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[0, :], points[1, :], points[2, :], s=1, c=points[2, :], cmap='viridis')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_box_aspect([1, 1, 1])

    if look_down == "x":
        ax.view_init(elev=0, azim=90)
    elif look_down == "y":
        ax.view_init(elev=0, azim=0)
    elif look_down == "z":
        ax.view_init(elev=90, azim=-90)

    if not axis_on:
        ax.set_axis_off()

    # set_axes_equal(ax)

def matplotlib_axplot3d(ax, points, axis_on=False, look_down="default"):
    def set_axes_equal(_ax):
        limits = np.array([
            _ax.get_xlim3d(),
            _ax.get_ylim3d(),
            _ax.get_zlim3d(),
        ])
        origin = np.mean(limits, axis=1)
        radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
        _set_axes_radius(_ax, origin, radius)

    def _set_axes_radius(_ax, origin, radius):
        x, y, z = origin
        _ax.set_xlim3d([x - radius, x + radius])
        _ax.set_ylim3d([y - radius, y + radius])
        _ax.set_zlim3d([z - radius, z + radius])

    ax.scatter(points[0, :], points[1, :], points[2, :], s=1, c=points[2, :], cmap='viridis')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_box_aspect([1, 1, 1])

    if look_down == "x":
        ax.view_init(elev=0, azim=90)
    elif look_down == "y":
        ax.view_init(elev=0, azim=0)
    elif look_down == "z":
        ax.view_init(elev=90, azim=-90)

    if not axis_on:
        ax.set_axis_off()

    set_axes_equal(ax)

def polyscope_plot3d(shape, color=None, quad_radius=0.001):
    ps.init()
    ps.set_ground_plane_mode("none")
    ps.register_point_cloud("pd", shape, point_render_mode="quad", radius=quad_radius)
    if color is not None:
        ps.get_point_cloud("pd").add_scalar_quantity("cl", color)
    ps.show()
