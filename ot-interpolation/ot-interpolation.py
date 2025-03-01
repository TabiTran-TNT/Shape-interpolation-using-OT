import numpy as np
import matplotlib.pyplot as plt
from generate_data import generate_point_cloud
from visualize import plot3d, axplot3d
from utils import distmat
from chamfer_distance import chamfer_distance, hausdorff_distance
from mesh_triangle import conformal_distortion


duck_points = generate_point_cloud("duck", 420)
print(duck_points.shape)
donut_points = generate_point_cloud("donut", 420)
print(donut_points.shape)

# plot3d(duck_points, False, "z")
# plot3d(donut_points, False, "z")
# plt.show()

a = np.ones(duck_points.shape[1])
b = np.ones(donut_points.shape[1])
C = distmat(duck_points, donut_points)
epsilon = .01
K = np.exp(-C/epsilon)
v = np.ones(donut_points.shape[1])
u = a / (np.dot(K, v))
v = b / (np.dot(np.transpose(K), u))
v = np.ones(donut_points.shape[1])

epoch_num = 50
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
    print("Epoch: ", i, " done.")


P = np.dot(np.dot(np.diag(u),K),np.diag(v))
plt.imshow(P)
I,J = np.nonzero(P > 1e-5)
Pij = P[I,J]
tlist = np.linspace(0, 1, 8)
Xtlist = []


fig, axes = plt.subplots(1, 8, figsize=(40, 5), subplot_kw={'projection': '3d'})
for i, ax in enumerate(axes.flat):
    Xt = (1 - tlist[i]) * duck_points[:, I] + tlist[i] * donut_points[:, J]
    Xtlist.append(Xt)
#     axplot3d(ax, Xt)
#     # plot3d(Xt, False, "z")
#     # plt.show()
#     print("Figure ", i, " done.")
# plt.show()


# print(Xtlist[0].shape)
# for i in range(7):
#     print(chamfer_distance(Xtlist[i].T, Xtlist[i + 1].T))
#
# # Seem to make sense
# print(chamfer_distance(Xtlist[0].T, Xtlist[7].T))
#
#
# print(Xtlist[0].shape)
# for i in range(7):
#     print(hausdorff_distance(Xtlist[i].T, Xtlist[i + 1].T))
#
# # Seem to make sense
# print(hausdorff_distance(Xtlist[0].T, Xtlist[7].T))


# Do not make sense
# res_lst = []
# print(Xtlist[0].shape)
# for i in range(7):
#     res_lst.append(conformal_distortion(Xtlist[i].T, Xtlist[i + 1].T))
#
#
# # Seem to make sense
# print(conformal_distortion(Xtlist[0].T, Xtlist[7].T))
# print(res_lst)


# import cvxpy as cp
#
# a = np.ones((duck_points.shape[1], 1))/duck_points.shape[1]
# b = np.ones((donut_points.shape[1], 1))/donut_points.shape[1]
#
# def distmat(x,y):
#     return np.sum(x**2,0)[:,None] + np.sum(y**2,0)[None,:] - 2*x.transpose().dot(y)
#
# C = distmat(duck_points, donut_points)
# P = cp.Variable((duck_points.shape[1],donut_points.shape[1]))
#
# u = np.ones((donut_points.shape[1], 1))
# v = np.ones((duck_points.shape[1], 1))
# U = [0 <= P, cp.matmul(P,u)==a, cp.matmul(P.T,v)==b]
#
# objective = cp.Minimize( cp.sum(cp.multiply(P,C)) )
# prob = cp.Problem(objective, U)
# result = prob.solve()
