import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cvxpy as cp

def distmat(x, y):
    return np.sum(x**2, 0)[:, None] + np.sum(y**2, 0)[None, :] - 2*x.transpose().dot(y)

def axplot3d(ax, points, axis_on=False, look_down="default"):
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

    if look_down == "y":
      ax.view_init(elev=0, azim=90)
    elif look_down == "x":
      ax.view_init(elev=0, azim=0)
    elif look_down == "z":
      ax.view_init(elev=90, azim=-90)
    elif look_down == "default":
      ax.view_init(elev=30, azim=-30)

    if not axis_on:
        ax.set_axis_off()
    set_axes_equal(ax)


def sinkhorn():
    a = np.array(Image.open("compare-1.png"))
    b = np.array(Image.open("compare-2.png"))

    a = a.flatten()
    b = b.flatten()
    print(a.shape)

    x = np.array(range(0, 32))
    y = np.array(range(0, 32))
    (x, y) = np.meshgrid(x, y)
    image_1 = np.vstack([y.ravel(), x.ravel()])
    image_2 = np.vstack([y.ravel(), x.ravel()])

    C = distmat(image_1, image_2)
    epsilon = 0.4
    K = np.exp(-C/epsilon)
    v = np.ones(image_1.shape[1])
    u = a / (np.dot(K, v))
    v = b / (np.dot(np.transpose(K), u))

    epoch_num = 500
    Err_p = []
    Err_q = []

    for i in range(epoch_num):
        # sinkhorn step 1
        u = a / (np.dot(K, v))
        # error computation
        r = v*np.dot(np.transpose(K), u)
        Err_q = Err_q + [np.linalg.norm(r - b, 1)]
        # sinkhorn step 2
        v = b / (np.dot(np.transpose(K), u))
        s = u*np.dot(K, v)
        Err_p = Err_p + [np.linalg.norm(s - a, 1)]


    P = np.dot(np.dot(np.diag(u),K),np.diag(v))
    plt.imshow(P)
    I,J = np.nonzero(P > 1e-5)
    Pij = P[I,J]
    tlist = np.linspace(0, 1, 9)

    fig, axes = plt.subplots(1, 9, figsize=(45, 5))
    # fig.subplots_adjust(hspace=0)
    fig.subplots_adjust(wspace=0.05)
    print(list(enumerate(axes.flat)))
    for i, ax in enumerate(axes.flat):
        temp_image = np.zeros((32, 32))
        Xt = (1 - tlist[i]) * image_1[:, I] + tlist[i] * image_2[:, J]
        count = 0
        for e in Xt.T:
            temp_image[int(e[0]), int(e[1])] += Pij[count]
            count += 1
        temp_image = temp_image ** 0.4
        ax.imshow(temp_image, 'gray')

        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])

        if i % 9 == 0 or i % 9 == 8:
            frame_color = 'red'
            ax.spines['top'].set_color(frame_color)
            ax.spines['bottom'].set_color(frame_color)
            ax.spines['left'].set_color(frame_color)
            ax.spines['right'].set_color(frame_color)

            frame_width = 3  # Increase the width
            ax.spines['top'].set_linewidth(frame_width)
            ax.spines['bottom'].set_linewidth(frame_width)
            ax.spines['left'].set_linewidth(frame_width)
            ax.spines['right'].set_linewidth(frame_width)
    fig.savefig("sinkhorn-2d.png", bbox_inches='tight')
    plt.show()


def solver():
    a = np.array(Image.open("compare-1.png"))
    b = np.array(Image.open("compare-2.png"))

    a = a.flatten().reshape(-1, 1)
    a = a / np.sum(a)
    b = b.flatten().reshape(-1, 1)
    b = b / np.sum(b)

    x = np.array(range(0, 32))
    y = np.array(range(0, 32))
    (x, y) = np.meshgrid(x, y)
    image_1 = np.vstack([y.ravel(), x.ravel()])
    image_2 = np.vstack([y.ravel(), x.ravel()])

    C = distmat(image_1, image_2)
    P = cp.Variable((image_1.shape[1],image_2.shape[1]))

    u = np.ones((image_1.shape[1], 1))
    v = np.ones((image_2.shape[1], 1))
    U = [0 <= P, cp.matmul(P,u)==a, cp.matmul(P.T,v)==b]

    objective = cp.Minimize( cp.sum(cp.multiply(P,C)) )
    prob = cp.Problem(objective, U)

    prob.solve(verbose=True, max_iter=100)

    P = P.value
    #plt.imshow(P)
    I,J = np.nonzero(P > 1e-5)
    Pij = P[I,J]
    tlist = np.linspace(0, 1, 9)

    fig, axes = plt.subplots(1, 9, figsize=(45, 5))
    # fig.subplots_adjust(hspace=0)
    fig.subplots_adjust(wspace=0.05)
    print(list(enumerate(axes.flat)))
    for i, ax in enumerate(axes.flat):
        temp_image = np.zeros((32, 32))
        Xt = (1 - tlist[i]) * image_1[:, I] + tlist[i] * image_2[:, J]
        count = 0
        for e in Xt.T:
            temp_image[int(e[0]), int(e[1])] += Pij[count]
            count += 1
        temp_image = temp_image ** 0.4
        ax.imshow(temp_image, 'gray')

        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])

        if i % 9 == 0 or i % 9 == 8:
            frame_color = 'red'
            ax.spines['top'].set_color(frame_color)
            ax.spines['bottom'].set_color(frame_color)
            ax.spines['left'].set_color(frame_color)
            ax.spines['right'].set_color(frame_color)

            frame_width = 3  # Increase the width
            ax.spines['top'].set_linewidth(frame_width)
            ax.spines['bottom'].set_linewidth(frame_width)
            ax.spines['left'].set_linewidth(frame_width)
            ax.spines['right'].set_linewidth(frame_width)
    fig.savefig("solver-2d.png", bbox_inches='tight')
    plt.show()

solver()
