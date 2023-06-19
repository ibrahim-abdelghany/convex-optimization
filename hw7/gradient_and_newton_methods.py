import numpy as np
from scipy.linalg import solve_triangular

import matplotlib.pyplot as plt

n = 100
m = 200

random_state = np.random.RandomState(1)

A = random_state.rand(m,n)

def dom(x):
    return np.all(A @ x < 1) and np.all(np.abs(x) < 1)
def f(x):
    return - np.sum(np.log(1 - A @ x)) - np.sum(np.log(1 - np.square(x)))
def gradient(x):
    return (1 / (1 - A @ x)) @ A + 2 * x / (1 - np.square(x))
def hessian(x):
    return A.T @ np.diag(1 / np.square(1 - A @ x)) @ A \
        + 2 * np.diag(1 / (1 - np.square(x))) \
        + 4 * np.diag(np.square(x)/np.square(1 - np.square(x)))

def main():
    run_approximation_newton()

def run_approximation_newton():
    x0 = np.zeros(n)

    alpha = 0.3
    beta = 0.8

    top_left, top_middle, top_right = plt.subplot(2, 3, 1), plt.subplot(2, 3, 2), plt.subplot(2, 3, 3)
    bottom_left, bottom_middle, bottom_right = plt.subplot(2, 3, 4), plt.subplot(2, 3, 5), plt.subplot(2, 3, 6)

    x_opt = newton(x0, f, dom, gradient, hessian, 1e-10, alpha, beta)

    steps, error = get_stats(x_opt, f)

    flops = np.zeros(len(x_opt)) + 1/3 * n**3 + 2 * n**2
    cum_flops = np.cumsum(flops)

    top_left.plot(cum_flops[1:], steps)
    top_left.set_yscale('log')
    top_left.set_ylabel('step size')
    top_left.set_title('full newton')

    bottom_left.plot(cum_flops[:-1], error)
    bottom_left.set_yscale('log')
    bottom_left.set_xlabel('flops')
    bottom_left.set_ylabel('error')

    reuse = 15

    x_opt = newton(x0, f, dom, gradient, hessian, 1e-10, alpha, beta, reuse=reuse)

    steps, error = get_stats(x_opt, f)

    flops = np.fromfunction(lambda i: np.heaviside(-(i % reuse), 1) * 1/3 * n**3, (len(x_opt),)) + 2 * n**2
    cum_flops = np.cumsum(flops)

    top_middle.plot(cum_flops[1:], steps)
    top_middle.set_yscale('log')
    top_middle.set_title('re-use hessian 15 times')

    bottom_middle.plot(cum_flops[:-1], error)
    bottom_middle.set_yscale('log')
    bottom_middle.set_xlabel('flops')

    x_opt = newton(x0, f, dom, gradient, hessian, 1e-10, alpha, beta, diag=True)

    steps, error = get_stats(x_opt, f)

    flops = np.zeros(len(x_opt)) + n
    cum_flops = np.cumsum(flops)

    top_right.plot(cum_flops[1:], steps)
    top_right.set_yscale('log')
    top_right.set_title('use hessian diagonal')

    bottom_right.plot(cum_flops[:-1], error)
    bottom_right.set_yscale('log')
    bottom_right.set_xlabel('flops')

    plt.legend()

    plt.show()

def run_gradient_newton():
    x0 = np.zeros(n)
    tol = 1e-5

    alpha = 0.3
    beta = 0.8

    x_opt = gradient_descent(x0, f, dom, gradient, tol, alpha, beta)
    
    iterations_steps, steps, iterations_error, error = get_stats(x_opt, f)
    
    top_left, bottom_left = plt.subplot(2, 2, 1), plt.subplot(2, 2, 2)
    top_right, bottom_right = plt.subplot(2, 2, 3), plt.subplot(2, 2, 4)

    top_left.set_label('Gradient Descent step size')
    top_left.plot(iterations_steps, steps, label='step size')
    top_left.set_yscale('log')
    
    bottom_left.set_label('Gradient Descent error')
    bottom_left.plot(iterations_error, error, label='f(x) - p*')
    bottom_left.set_yscale('log')

    x_opt = newton(x0, f, dom, gradient, hessian, 1e-10, alpha, beta)

    iterations_steps, steps, iterations_error, error = get_stats(x_opt, f)

    top_left.set_label('Newtons Method step size')
    top_right.plot(iterations_steps, steps, label='step size')
    top_right.set_yscale('log')

    top_left.set_label('Newtons Method error')
    bottom_right.plot(iterations_error, error, label='f(x) - p*')
    bottom_right.set_yscale('log')

    plt.legend()

    plt.show()

def get_stats(x_opt, f):
    p_opt = list(map(f, x_opt))
    steps = np.linalg.norm(np.array(x_opt[1:]) - np.array(x_opt[:-1]), axis=1)
    error = np.array(p_opt[:-1]) - p_opt[-1]
    return steps, error

def newton(x0, f, dom, gradient, hessian, tol, alpha, beta, reuse=1, diag=False):
    x = [x0]

    i = 0
    L = None
    d = None

    while True:        
        grad = gradient(x[-1])

        if i % reuse == 0:
            if diag:
                d = np.diagonal(hessian(x[-1]))
            else:
                L = np.linalg.cholesky(hessian(x[-1]))
        
        if diag:
            newton_step = - grad/d
        else:
            newton_step = - solve_triangular(L.T, solve_triangular(L, grad, lower=True))
        
        newton_decrement = - grad @ newton_step

        if newton_decrement/2 <= tol:
            return x
        
        x.append(backtracking_line_search(f, dom, gradient, x[-1], newton_step, alpha, beta))
        
        i+=1

def gradient_descent(x0, f, dom, gradient, tol, alpha, beta):
    x = [x0]
    while np.linalg.norm(gradient(x[-1])) > tol:
        delta = - gradient(x[-1])
        x.append(backtracking_line_search(f, dom, gradient, x[-1], delta, alpha, beta))
    return x

def backtracking_line_search(f, dom, gradient, x, delta, alpha, beta):
    t = 1
    
    # domain is a convex set => if x+t*delta in dom => (x,x+t*delta) in dom
    while not dom(x + t * delta):
        t = beta * t
    
    while f(x + t * delta) > f(x) + alpha * t * gradient(x) @ delta:
        t = beta * t

    return x + t * delta

if __name__ == '__main__':
    main()