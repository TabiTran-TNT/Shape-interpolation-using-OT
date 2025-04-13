import sys
sys.path.append("./src")

import numpy as np
from scipy.special import logsumexp
from cost_matrix import calculate_cost_matrix, calculate_kernel
import ot


def ot_multiple_shapes_interpolate(shape_lst, steps=5):
    n = len(shape_lst)
    vals = np.linspace(0, 1, steps)
    xss = list(np.meshgrid(*([vals]*n)))
    for i in range(len(xss)):
        xss[i] = xss[i].flatten()
    lambda_lst_lst = list(zip(*xss))
    lambda_lst_lst.sort(reverse=True)

    barycenter_lst = []
    returned_lambda_lst_lst = []
    for lambda_lst in lambda_lst_lst:
        if sum(lambda_lst) == 1.0:
            barycenter = compute_barycenter(shape_lst, lambda_lst, 0.00001, 10)
            barycenter_lst.append(barycenter)
    return barycenter_lst, returned_lambda_lst_lst


def ot_shape_interpolate(source:np.ndarray, target:np.ndarray, steps=5):
    p_matrix = ot.sinkhorn(np.ones(source.shape[0]),
                           np.ones(target.shape[0]),
                           calculate_cost_matrix(source, target),
                           reg=0.02,
                           method="sinkhorn",
                           numItermax=10000,
                           verbose=True)

    i_lst, j_lst = np.nonzero(p_matrix > 1e-5)
    tlist = np.linspace(0, 1, steps)

    shape_lst = []
    for t in tlist:
        shape_t = (1 - t) * source[i_lst, :] + t * target[j_lst, :]
        shape_lst.append(shape_t)

    return shape_lst


def sinkhorn(source:np.ndarray, target:np.ndarray, epsilon=0.01, epoch_num=100):
    a = np.ones(source.shape[0])
    b = np.ones(target.shape[0])
    matrix_c = calculate_cost_matrix(source, target, kind="l2")
    matrix_k = np.exp(-matrix_c / epsilon)
    u = np.ones(source.shape[0])
    v = b / np.dot(np.transpose(matrix_k), u)

    for _ in range(epoch_num):
        u = a / np.dot(matrix_k, v)
        v = b / np.dot(np.transpose(matrix_k), u)
        print("Interation ", _)

    matrix_p = np.dot(np.dot(np.diag(u), matrix_k), np.diag(v))

    return matrix_p


def sinkhorn_log(source: np.ndarray, target: np.ndarray, epsilon=0.01, epoch_num=10):
    a = np.ones(source.shape[0])
    b = np.ones(target.shape[0])
    matrix_c = calculate_cost_matrix(source, target, kind="l2")

    u = np.zeros(source.shape[0])
    v = np.zeros(target.shape[0])

    for _ in range(epoch_num):
        u = epsilon * np.log(a) - epsilon * logsumexp((v[None, :] - matrix_c) / epsilon, axis=1)
        v = epsilon * np.log(b) - epsilon * logsumexp((u[:, None] - matrix_c) / epsilon, axis=0)

    log_matrix_p = (u[:, None] + v[None, :] - matrix_c) / epsilon
    matrix_p = np.exp(log_matrix_p)

    return matrix_p


def solver():
    pass
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


def compute_barycenter(shape_lst, lambda_lst, epsilon=0.0001, epoch_num=100):
    dimension = list(shape_lst[0].shape)
    dimension.append(len(shape_lst))
    print(dimension)
    prev_barycenter = None
    v_tensor = np.ones(dimension)
    u_tensor = np.copy(v_tensor)

    for _ in range(epoch_num):
        for i in range(len(shape_lst)):
            matrix_k = calculate_kernel(v_tensor[..., i], epsilon)
            u_tensor[..., i] = shape_lst[i] / matrix_k

        log_barycenter = np.zeros(dimension[:-1])
        for i in range(len(shape_lst)):
            matrix_k = calculate_kernel(u_tensor[..., i], epsilon)
            log_barycenter = (log_barycenter + lambda_lst[i]
                              * np.log(np.maximum(1e-19 * np.ones(len(v_tensor[..., i])),
                                                  v_tensor[..., i] * matrix_k)))
        barycenter = np.exp(log_barycenter)
        print(np.min(barycenter))
        print(np.max(barycenter))
        if prev_barycenter is None:
            prev_barycenter = barycenter
        else:
            print(np.linalg.norm(barycenter - prev_barycenter))

        for i in range(len(shape_lst)):
            matrix_k = calculate_kernel(u_tensor[..., i], epsilon)
            v_tensor[..., i] = barycenter / matrix_k

        print("Done epoch ", _)

    log_barycenter = np.zeros(dimension[:-1])
    for i in range(len(shape_lst)):
        matrix_k = calculate_kernel(u_tensor[..., i], epsilon)
        log_barycenter = (log_barycenter + lambda_lst[i]
                          * np.log(np.maximum(1e-19 * np.ones(len(v_tensor[..., i])),
                                              v_tensor[..., i] * matrix_k)))
    return np.exp(log_barycenter)
