import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

from piecewise_linear_data import x,y

### EE364a Homework 6 additional problems
# Exercise 3

def main():
    plt.plot(x, y, 'r', label='data')

    for knots in range(0,4):
        a,b = fit_lagrange_basis(x,y, knots)

        plt.plot(a, b, label=str(knots) + ' knots')

        for (alpha, beta) in get_linear_pieces(a, b):
            print(alpha , '+', beta, ' * x')
    
    plt.legend()
    plt.show()

def get_linear_pieces(a,b):
    linear_pieces = []
    # TODO find vectorized or functional alternative
    for i in range(len(a)-1):
        alpha = b[i] - a[i] * (b[i+1] - b[i])/(a[i+1] - a[i])
        beta = (b[i+1] - b[i])/(a[i+1] - a[i])
        linear_pieces.append((alpha, beta))
    return linear_pieces

def fit_lagrange_basis(x, y, knots):
    n = len(x)
    k = knots + 2
    
    a = np.linspace(0, 1, k)
    b = cp.Variable(k)
    h = cp.Variable(k)
    
    # TODO find functional or vectorized approach to compute g
    g = np.zeros((n,k))

    for i in range(n):
        for j in range(k):
            if j > 0 and a[j-1] <= x[i] and x[i] <= a[j]:
                g[i,j] = (x[i] - a[j-1]) / (a[j] - a[j-1])
            elif j < k and a[j] <= x[i] and x[i] <= a[j+1]:
                g[i,j] = (a[j+1] - x[i]) / (a[j+1] - a[j]) 

    objective = cp.sum_squares(g @ b - y)

    constraints = [ 
        b >= b[i] + h[i] * (a - a[i]) for i in range(k)
    ]

    problem = cp.Problem(cp.Minimize(objective), constraints)

    result = problem.solve()

    print('result', result)

    return (a, b.value)

if __name__ == '__main__':
    main()