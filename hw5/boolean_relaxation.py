import functools
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

### EE364a Homework 5 additional problems
# Exercise 6

n = 100
m = 300
T = 100

def main():
    print('Generating random parameters')
    A = np.random.random((m,n))
    b = A @ np.ones(n) / 2
    c = - np.random.random(n)

    print('Finding optimal relaxed solution')
    opt_x_rlx, opt_result_rlx = optimal_relaxed(A, b, c)

    print('Relaxation lower-bound', opt_result_rlx)

    print('Generating boolean solutions for different thresholds')
    t = np.linspace(0, 1, T)

    # generate boolean vectors x for each threshold value
    t_x_bools = np.heaviside(opt_x_rlx - np.repeat(np.reshape(t, (T,1)), n, 1), 1)

    # generate max constrarint violation
    t_constraint_penalty = np.max(t_x_bools @ np.transpose(A) - b, axis=1)

    # generate constraint irritation (violation -> infinity)
    t_constraint_irritation = np.where(t_constraint_penalty > 0, np.inf, 0)

    t_upper_bound = t_x_bools @ c # + t_constraint_irritation

    t_min_index = np.argmin(t_upper_bound + t_constraint_irritation)
    t_min, t_min_upper_bound = t[t_min_index], t_upper_bound[t_min_index]

    print('Minimum t: ', t_min, 'Minimum upper bound:', t_min_upper_bound, '')

    plt.figure('Boolean Relaxation')

    top_plot = plt.subplot(2,1,1)
    # top_plot.yaxis.label = 'Constraint violation'
    bottom_plot = plt.subplot(2,1,2)
    # bottom_plot.yaxis.label = 'Objective'

    top_plot.plot(t, np.repeat(0,(T)), 'k', label='Violation Threshold')
    top_plot.plot(t, np.ma.masked_where(t_constraint_penalty >= 0, t_constraint_penalty), 'g', label='Constraints satisfied')
    top_plot.plot(t, np.ma.masked_where(t_constraint_penalty < 0, t_constraint_penalty), 'r', label='Constraints violated')
    top_plot.set_ylabel('Constraint Violation')
    top_plot.legend()

    # plot the lower-bound
    bottom_plot.plot(t, np.repeat(opt_result_rlx, T), 'b', label='Lower-bound')
    bottom_plot.plot(t, np.ma.masked_where(t_constraint_penalty >= 0, t_upper_bound), 'g', label='Upper-bound')
    bottom_plot.plot(t, np.ma.masked_where(t_constraint_penalty < 0, t_upper_bound), 'r')
    bottom_plot.set_ylabel('Objective')
    bottom_plot.set_xlabel('Threshold')
    bottom_plot.legend()
    
    plt.show()

def optimal_relaxed(A, b, c):
    x = cp.Variable(n)

    constraints = [
        A @ x <= b,
        x >= 0,
        x <= 1
    ]

    objective = c @ x

    problem = cp.Problem(cp.Minimize(objective), constraints)

    result = problem.solve()

    return (x.value, result)

if __name__ == '__main__':
    main()