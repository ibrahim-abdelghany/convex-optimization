from functools import partial

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.linalg import pinv

import cvxpy as cp

import matplotlib.pyplot as plt

n = 500
m = 100

random_state = np.random.RandomState(1)

A = np.append(random_state.rand(m-1, n) - 0.5, [random_state.rand(n)], axis=0)
A_infeasible = random_state.rand(m, n) - 1
c = random_state.rand(n) - 0.5
x0 = random_state.rand(n)
b = A @ x0

alpha = 0.3
beta = 0.8

t0 = 30
mu = 10

def main():
    main_phase_1_phase_2()

def main_phase_1_phase_2():
    feasible, x0 = phase_1(A, b, t0, mu, alpha, beta)
    print('Is feasible?', feasible)
    cvx_result, cvx_opt = cvx_solve_LP(A, b, c)
    cvx_feasible = cvx_result < np.inf
    print('cvx is feasible?', cvx_feasible)

    x_opt, history = feasible_start_barrier(A, c, t0, x0, mu, alpha, beta)

    assert np.allclose(cvx_opt, x_opt, rtol=1e-2, atol=1e-2)

def main_feasible_start():
    res_cvx, x_opt_cvx = cvx_solve_LP(A, b, c)

    x, history = feasible_start_barrier(A, c, t0, x0, mu, alpha, beta)
    
    assert np.allclose(x_opt_cvx, x, atol=1e-3)

    plt.stairs(np.cumsum(history[:,0]), np.concatenate([[m], history[:,1]]), orientation='horizontal')
    plt.yscale('log')
    plt.show()

def main_centering():
    
    assert np.allclose(
            np.concatenate(solve_kkt_system_block_elimination(A, c, x0)), 
            np.concatenate(solve_kkt_system_naive(A, c, x0))
        )

    x, nu, progress = centering_LP(A, c, x0, alpha, beta)

    plt.plot(progress['lambda'], label='newton decrement')
    plt.yscale('log')
    plt.show()

def phase_1(A, b, t_barrier, mu, alpha, beta):
    m, n = A.shape

    A_1 = np.column_stack((A, - A @ np.ones(n)))
    b_1 = b - A @ np.ones(n)

    c_1 = np.concatenate([np.zeros(n), [1]])

    x0 = pinv(A) @ b
    t0 = 2 - min(x0)

    z0 = np.concatenate([x0 + np.ones(n) * (t0-1), [t0]])

    z_opt, progress = feasible_start_barrier(A_1, c_1, t_barrier, z0, mu, alpha, beta)

    return z_opt[-1] < 1, z_opt[:-1] - (z_opt[-1] - 1) * np.ones(n)

def cvx_solve_LP(A, b, c):
    m, n = A.shape

    x = cp.Variable(n)

    constraints = [
        x >= 0,
        A @ x == b
    ]

    result = cp.Problem(cp.Minimize(c @ x), constraints).solve()

    return (result, x.value)

def feasible_start_barrier(A, c, t0, x0, mu, alpha, beta):
    t = t0
    x_opt = x0

    progress = []

    m, n = A.shape
    
    while True:
        x_opt, nu_opt, centering_progress = centering_LP(A, t * c, x_opt, alpha, beta)
        progress.append([len(centering_progress), m/t])
        if m/t <= 1e-3:
            return x_opt, np.array(progress)
        t = mu * t

def f(c, x):
    return c.T @ x - np.sum(np.log(x))

def dom(x):
    return np.all(x > 0)

def gradient(c, x):
    return c - 1/x

def centering_LP(A, c, x0, alpha, beta, naive=False):

    kkt_solver = partial(solve_kkt_system_naive if naive else solve_kkt_system_block_elimination, A, c)
    
    progress = {
        'lambda': [],
        'x': [x0]
    }

    x = x0
    nu = 0

    while True:
        delta_x_nt, nu_, grad = kkt_solver(x)
        newton_decrement = - grad @ delta_x_nt

        progress['lambda'].append(newton_decrement)

        if newton_decrement <= 1e-6:
            return x, nu, progress
        
        t = backtracking_line_search(partial(f, c), dom, grad, x, delta_x_nt, alpha, beta)

        x = x + t * delta_x_nt
        nu = nu_

        progress['x'].append(x)

"""
KKT_Matrix = [
    [diag(1/x^2),   A.T],
    [A,             0]
]
KKT System:
KKT_matrix @ [delta_x_nt, nu] = - [gradient(x), 0]
"""
def solve_kkt_system_naive(A, c, x):
    m, n = A.shape
    grad = gradient(c, x)
    KKT_matrix = np.row_stack([
        np.column_stack([np.diag(1/np.square(x)), A.T]), 
        np.column_stack([A, np.zeros((m,m))])
    ])
    y = - solve(KKT_matrix, np.concatenate([grad, np.zeros(m)]))
    return (y[:n], y[n:], grad)

"""
KKT_Matrix = [
    [diag(1/x^2),   A.T],
    [A,             0]
]
KKT System:
KKT_matrix @ [delta_x_nt, nu] = - [gradient(x), 0]

Block Elimination:
delta_x_nt = inv(diag(1/x^2)) @ (-gradient(x) - A.T @ nu)
nu = - inv((A @ inv(diag(1/x^2)) @ A.T)) @ ( A @ inv(diag(1/x^2)) @ gradient(x))
"""
def solve_kkt_system_block_elimination(A, c, x):
    grad = gradient(c, x)
    A_diag_inv = A @ np.diag(np.square(x))
    nu = - cho_solve(cho_factor(A_diag_inv @ A.T), A_diag_inv @ grad)
    delta_x_nt = np.diag(np.square(x)) @ (-grad - A.T @ nu)
    return (delta_x_nt, nu, grad)

def backtracking_line_search(f, dom, grad, x, delta, alpha, beta):
    t = 1
    
    # domain is a convex set => if x+t*delta in dom => (x,x+t*delta) in dom
    while not dom(x + t * delta):
        t = beta * t
    
    while f(x + t * delta) > f(x) + alpha * t * grad @ delta:
        t = beta * t

    return t

if __name__ == '__main__':
    main()
