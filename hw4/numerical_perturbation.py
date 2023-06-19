import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

### EE364a Homework 4 additional problems
# Exercise 3

def main():
    opt, primal, dual = quadratic(-2, -3)

    perturbations = np.array([[[delta1,delta2] for delta2 in np.linspace(-0.1,0.1,3)] for delta1 in np.linspace(-0.1,0.1,3)]).reshape(9,2)
    print('perturbations:')
    print(perturbations)

    p_star_pred = opt - perturbations @ dual[:2]
    print('p_star_pred:')
    print(p_star_pred)

    p_star_exact = np.array([[quadratic(-2 + delta1, -3 + delta2)[0] for delta2 in np.linspace(-0.1,0.1,3)] for delta1 in np.linspace(-0.1,0.1,3)]).reshape(9)
    print('p_star_exact:')
    print(p_star_exact)

    assert np.all(p_star_exact >= p_star_pred)

def quadratic(u1, u2):
    x = cp.Variable(2)

    constraints = [
        np.array([1, 2]) @ x <= u1,
        np.array([1, -4]) @ x <= u2,
        np.array([5, 76]) @ x <= 1
    ]

    objective = cp.quad_form(x, np.array([[1, -0.5], [-0.5, 2]])) - x[0]

    problem = cp.Problem(cp.Minimize(objective), constraints)

    result = problem.solve()

    return (result, x.value, [float(constraints[i].dual_value) for i in range(3)])

if __name__ == '__main__':
    main()