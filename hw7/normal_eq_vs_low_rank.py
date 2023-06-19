import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.linalg import solve_triangular, cholesky

n = 2000
k = 100
eta = delta = 1

random_state = np.random.RandomState(1)

A = random_state.rand(k, n)

b = random_state.rand(k)

D = delta * sparse.diags([-np.ones(n-1), np.concatenate([[1], 2* np.ones(n-2), [1]]), -np.ones(n-1)], [-1, 0, 1])\
+ eta * sparse.identity(n)

def main():
    x1 = normal_solver(A, b, D)
    print(x1)

    x2 = low_rank_solver(A, b, D)
    print(x2)

    assert np.allclose(x1, x2)

def low_rank_solver(A, b, D):
    D_inv_AT = spsolve(D.tocsc(), A.T)

    w = solve_cholesky(sparse.identity(k) + A @ D_inv_AT, 
                       A @ D_inv_AT @ b)

    x = D_inv_AT @ (b - w)

    return x

def normal_solver(A, b, D):
    return solve_cholesky(A.T @ A + D, A.T @ b)

def solve_cholesky(S, b):
    L = cholesky(S, lower=True)
    return solve_triangular(L.T, solve_triangular(L, b, lower=True))


if __name__ == '__main__':
    main()