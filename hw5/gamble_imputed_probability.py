import functools
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

### EE364a Homework 5 additional problems
# Exercise 5

n = 5
m = 5

def main():
    p = np.array([0.5, 0.6, 0.6, 0.6, 0.2])
    q = np.array([10, 5, 5, 20, 10])
    S = np.array([
        [1, 0, 1, 0, 0],
        [1, 0, 0, 1, 0],
        [0, 0, 0, 0, 1],
        [0, 1, 1, 0, 0],
        [0, 0, 1, 1, 0],
        ])

    profit, probabilities = optimal_worst_case_profit(p, q, S)

    print("optimal worst-case profit", profit) 
    print("imputed probabilities", probabilities)

    worst_case_profit = q @ p - max(S @ q)

    print("worst-case profit", worst_case_profit)

def optimal_worst_case_profit(p, q, S):
    x = cp.Variable(n)
    t = cp.Variable()

    constraints = [
        q >= x,
        x >= 0,
        S @ x <= np.ones(n) * t
    ]

    objective = p @ x - t

    problem = cp.Problem(cp.Maximize(objective), constraints)

    result = problem.solve()

    return (result, constraints[2].dual_value)

if __name__ == '__main__':
    main()