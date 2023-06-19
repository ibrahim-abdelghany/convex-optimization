import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

### EE364a Homework 6 additional problems
# Exercise 1

n = 3
m = 3

def main():
    t = np.linspace(-3,3, 201)
    y = np.exp(t)

    a, b = fit_fractional(t, y, 0, 9, 0.001)

    print('a', a)
    print('b', b)
 
    plt.figure('Rational function fitting')

    top_plot = plt.subplot(2,1,1)
    bottom_plot = plt.subplot(2,1,2)

    top_plot.plot(t, y, 'b', label='Exponential')

    top_plot.plot(t, fractional(a, b, t), 'g', label='Rational')

    bottom_plot.plot(t, fractional(a, b, t) - y, 'r', label='Maximum error')

    plt.show()

def fit_fractional(t, y, lower, upper, tol):
    print('fitting..')

    last_a = None
    last_b = None

    while upper - lower >= tol:
        print('Fitting range', (lower, upper))
        alpha = lower + (upper-lower)/2

        a = cp.Variable(n)
        b = cp.Variable(m - 1)
        b_1 = cp.reshape(cp.vstack([*b, cp.Constant(1)]), (m,))
        
        numerator, denominator = polynomials(a, b_1, t)

        constraints = [
            numerator - cp.multiply(y,  denominator) <= alpha * denominator,
            numerator - cp.multiply(y, denominator) >= - alpha * denominator,
            denominator >= 0
        ]

        problem = cp.Problem(cp.Minimize(0), constraints)

        result = problem.solve()

        print('result', result)

        if result == 0:
            last_a = a.value
            last_b = b_1.value
            upper = alpha
        else:
            lower = alpha

    return (last_a, last_b)

def polynomials(a, b, t):
    t_poly_n = powers(t, n)
    t_poly_m = powers(t, m)

    numerator = t_poly_n @ a
    denominator = t_poly_m @ b

    return (numerator, denominator)

def fractional(a, b, t):
    numerator, denominator = polynomials(a, b, t)
    return numerator / denominator

def powers(t, degree):
    return np.power(np.transpose(np.repeat([t], degree, axis=0)), range(degree))


if __name__ == '__main__':
    main()