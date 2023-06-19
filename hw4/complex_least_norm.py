import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

### EE364a Homework 4 additional problems
# Exercise 2

def main():
    complex_least_norm_2(100, 30, 2)
    complex_least_norm_2(100, 30, np.Inf)

def complex_least_norm_2(n, m, l):
    A = np.random.randn(m, n) + 1j*np.random.randn(m, n)
    b = np.random.randn(m) + 1j*np.random.randn(m)
    x = cp.Variable(n, complex=True)

    least_norm = cp.Problem(cp.Minimize(cp.norm(x, l)), [A@x == b])

    result = least_norm.solve()

    # print('x=', x.value)
    plt.scatter(x.value.real, x.value.imag)
    plt.show()
    print('obj=', result)

if __name__ == '__main__':
    main()