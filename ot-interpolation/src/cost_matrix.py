import numpy as np


def calculate_kernel(v_matrix, epsilon=0.0001, kind="gaussian"):
    if kind == "gaussian":
        if len(v_matrix.shape) == 1:
            vals = np.linspace(0, 1, v_matrix.shape[0])
            [ys, xs] = np.meshgrid(vals, vals)
            matrix_h = np.exp(-(xs - ys) ** 2 / epsilon)
            return np.dot(matrix_h, v_matrix)
        elif len(v_matrix.shape) == 2:
            matrix_k = v_matrix.copy()
            for i in range(2):
                vals = np.linspace(0, 1, v_matrix.shape[i])
                [ys, xs] = np.meshgrid(vals, vals)
                matrix_h = np.exp(-(xs - ys) ** 2 / epsilon)
                matrix_k = np.dot(matrix_h, matrix_k)
                matrix_k = np.transpose(matrix_k, [1, 0])
            return matrix_k
        elif len(v_matrix.shape) == 3:
            matrix_k = v_matrix.copy()
            for i in range(3):
                vals = np.linspace(0, 1, v_matrix.shape[i])
                [ys, xs] = np.meshgrid(vals, vals)
                matrix_h = np.exp(-(xs - ys) ** 2 / epsilon)
                # print(matrix_h)
                matrix_k = np.dot(matrix_h, matrix_k)
                matrix_k = np.transpose(matrix_k, [1, 2, 0])
            return matrix_k
        else:
            return np.zeros(v_matrix.shape)
    else:
        return np.zeros(v_matrix.shape)


def calculate_cost_matrix(source:np.ndarray, target:np.ndarray, kind="l2"):
    if kind == "l2":
        c_matrix = np.zeros((source.shape[0], target.shape[0]))
        for i in range(target.shape[0]):
            c_matrix[:, i] = np.sqrt(np.linalg.norm(source - target[i], axis=1))
        return c_matrix
    else:
        return np.zeros((source.shape[0], target.shape[0]))
