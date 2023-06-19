import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

### EE364a Homework 4 additional problems
# Exercise 5

def main():
    A = np.array([[-1, 0.4, 0.8], [1, 0, 0], [0, 1, 0]])
    b = np.array([1, 0, 0.3])
    x_des = np.array([7, 2, -6])
    N = 30

    result, u = minimum_fuel(A, b, x_des, N)

    print(result)
   
    plt.stairs(u)
    plt.show()

def minimum_fuel(A, b, x_des, N):
    u = cp.Variable(N)

    constraints = [
        np.matmul([np.linalg.matrix_power(A, N-1-i) @ b for i in range(N)], u) == x_des
    ]

    objective = cp.Minimize(cp.sum(cp.maximum(cp.abs(u), 2*cp.abs(u)-1)))

    problem = cp.Problem(objective, constraints)
    
    result = problem.solve()

    return (result, u.value)

if __name__ == '__main__':
    main()