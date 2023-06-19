import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

from linear_separation_data import xs, ys, zs

def main():
    result, (a1, b1), (a2, b2), (a3, b3) = classify_3_way(xs, ys, zs)

    print(result)

    plt.scatter(np.array(xs).T[0], np.array(xs).T[1], 3, 'r')
    plt.scatter(np.array(ys).T[0], np.array(ys).T[1], 3, 'g')
    plt.scatter(np.array(zs).T[0], np.array(zs).T[1], 3, 'b')

    t = np.arange(-7, 7, 0.1)

    u1 = a1 - a2
    v1 = b1 - b2
    
    u2 = a2 - a3
    v2 = b2 - b3

    u3 = a3 - a1
    v3 = b3 - b1

    line1 = (-u1[0]*t + v1)/u1[1]
    idx1 = np.where(u2[0]*t + u2[1] * line1 - v2 > 0)

    plt.plot(t[idx1], line1[idx1], 'r')

    line2 = (-u2[0]*t + v2)/u2[1]
    idx2 = np.where(u3[0]*t + u3[1] * line2 - v3 > 0)

    plt.plot(t[idx2], line2[idx2], 'g')

    line3 = (-u3[0]*t + v3)/u3[1]
    idx3 = np.where(u1[0]*t + u1[1] * line3 - v1 > 0)

    plt.plot(t[idx3], line3[idx3], 'b')

    plt.xlim(-7, 7)
    plt.ylim(-7, 7)

    plt.show()


def classify_3_way(xs, yz, zs):
    a1 = cp.Variable(2)
    b1 = cp.Variable()

    a2 = cp.Variable(2)
    b2 = cp.Variable()

    a3 = cp.Variable(2)
    b3 = cp.Variable()

    constraints = [
        # xs separated from 
        a1.T @ xs - b1 >= a2.T @ xs - b2 + 1,
        a1.T @ xs - b1 >= a3.T @ xs - b3 + 1,

        a2.T @ ys - b2 >= a1.T @ ys - b1 + 1,
        a2.T @ ys - b2 >= a3.T @ ys - b3 + 1,

        a3.T @ zs - b3 >= a1.T @ zs - b1 + 1,
        a3.T @ zs - b3 >= a2.T @ zs - b2 + 1,
    ]

    result = cp.Problem(cp.Minimize(0), constraints).solve()

    return (result, (a1.value, b1.value), (a2.value, b2.value), (a3.value, b3.value))

if __name__ == '__main__':
    main()