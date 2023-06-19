import numpy as np
import cvxpy as cp

### EE364a Homework 6 additional problems
# Exercise 4

m = 4
n = 3

A_nom = np.array([
    [60, 45, 8],
    [90, 30, -30],
    [0, -8, -4],
    [30, 10, -10]
])

R = np.zeros((m,n)) + 0.05

b = np.array([-6, -3, 18, -9])

def main():
    res_ls, x_ls = nominal_least_squares(A_nom, b)

    print('Nominal error', res_ls)
    print('Nominal x', x_ls)

    res_rlsm, x_rlsm = robust_least_squares_minimalist(A_nom, R, b)

    print('Robust error', res_rlsm)
    print('Robust x', x_rlsm)

    res_rlsd, x_rlsd = robust_least_squares_lp_duality(A_nom, R, b)

    print('Robust duality error', res_rlsd)
    print('Robust duality x', x_rlsd)

def nominal_least_squares(A, b):
    x = cp.Variable(n)

    objective = cp.sum_squares(A @ x -b)

    problem = cp.Problem(cp.Minimize(objective))

    result = problem.solve()

    return (result, x.value)

def robust_least_squares_minimalist(A_nom, R, b):
    x = cp.Variable(n)

    objective = cp.sum_squares(cp.abs(A_nom @ x - b) + R @ cp.abs(x))

    problem = cp.Problem(cp.Minimize(objective))

    result = problem.solve()

    return (result, x.value)

def robust_least_squares_lp_duality(A_nom, R, b):
    x = cp.Variable(n)

    dual = cp.Variable(n)

    t = cp.Variable(m)

    l = cp.Variable(m)

    objective = cp.sum_squares(l)

    constraints = [
        cp.maximum(cp.abs(A_nom @ x - b + t), cp.abs(A_nom @ x - b - t)) <= l,
        -R @ x - 2 * R @ dual >= t,
        dual >= 0,
        x + dual >=0
    ]

    problem = cp.Problem(cp.Minimize(objective), constraints)

    result = problem.solve()

    return (result, x.value)

if __name__ == '__main__':
    main()