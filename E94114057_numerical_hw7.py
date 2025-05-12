import numpy as np

A = np.array([
    [4, -1, 0, -1, 0, 0],
    [-1, 4, -1, 0, -1, 0],
    [0, -1, 4, 0, 1, -1],
    [-1, 0, 0, 4, -1, -1],
    [0, -1, 0, -1, 4, -1],
    [0, 0, -1, 0, -1, 4]
], dtype=float)

b = np.array([0, -1, 9, 4, 8, 6], dtype=float)
x0 = np.zeros_like(b)

def jacobi(A, b, x0, tol=1e-10, max_iter=100):
    x = x0.copy()
    D = np.diag(A)
    R = A - np.diagflat(D)
    for _ in range(max_iter):
        x_new = (b - np.dot(R, x)) / D
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            return np.round(x_new, 6)
        x = x_new
    return np.round(x, 6)

def gauss_seidel(A, b, x0, tol=1e-10, max_iter=100):
    x = x0.copy()
    n = len(b)
    for _ in range(max_iter):
        x_new = x.copy()
        for i in range(n):
            s1 = np.dot(A[i, :i], x_new[:i])
            s2 = np.dot(A[i, i+1:], x[i+1:])
            x_new[i] = (b[i] - s1 - s2) / A[i, i]
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            return np.round(x_new, 6)
        x = x_new
    return np.round(x, 6)

def sor(A, b, x0, omega=1.1, tol=1e-10, max_iter=100):
    x = x0.copy()
    n = len(b)
    for _ in range(max_iter):
        x_new = x.copy()
        for i in range(n):
            s1 = np.dot(A[i, :i], x_new[:i])
            s2 = np.dot(A[i, i+1:], x[i+1:])
            x_new[i] = (1 - omega) * x[i] + omega * (b[i] - s1 - s2) / A[i, i]
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            return np.round(x_new, 6)
        x = x_new
    return np.round(x, 6)

def conjugate_gradient(A, b, x0, tol=1e-10, max_iter=100):
    x = x0.copy()
    r = b - A @ x
    p = r.copy()
    rs_old = np.dot(r, r)
    for _ in range(max_iter):
        Ap = A @ p
        alpha = rs_old / np.dot(p, Ap)
        x += alpha * p
        r -= alpha * Ap
        rs_new = np.dot(r, r)
        if np.sqrt(rs_new) < tol:
            return np.round(x, 6)
        p = r + (rs_new / rs_old) * p
        rs_old = rs_new
    return np.round(x, 6)

# 執行方法
x_jacobi = jacobi(A, b, x0)
x_gs = gauss_seidel(A, b, x0)
x_sor = sor(A, b, x0, omega=1.1)
x_cg = conjugate_gradient(A, b, x0)

# 輸出
print("\nJacobi Method Solution:", x_jacobi)
print("\nGauss-Seidel Method Solution:", x_gs)
print("\nSOR Method Solution:", x_sor)
print("\nConjugate Gradient Method Solution:", x_cg)
