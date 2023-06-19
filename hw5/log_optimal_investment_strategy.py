import functools
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

### EE364a Homework 5 additional problems
# Exercise 4

n = 5
m = 10

def main():
    print('Generating random prices')
    prob = np.repeat(1/m, m)
    prices = np.random.normal(1, 0.05, (m,n))

    print('Generating strategies')
    uniform_strategy = np.repeat(1/n, n)
    optimal_strategy, growth_rate = log_optimal_strategy(prob, prices)

    print('Growth rate:', growth_rate)

    print('Simulating strategies')
    for j in range(10):
        uniform_trajectory = simulate_strategy(1, uniform_strategy, prob, prices, 200)
        optimal_trajectory = simulate_strategy(1, optimal_strategy, prob, prices, 200)

        print('Plotting results')
        x = [i for i in range(201)]
        
        plt.semilogy(x, uniform_trajectory, 'b--', x, optimal_trajectory, 'r--')
    plt.show()

def simulate_strategy(base, strategy, prob, prices, iterations):
    returns = [prices[sample_from_pmf(prob)] @ strategy for i in range(iterations)]

    return functools.reduce(lambda x, y: [*x, x[-1] * y], returns, [base])

def sample_from_pmf(prob):
    cumulative_pmf = [sum(prob[:i+1]) for i in range(len(prob))]

    uniform = np.random.random()

    return len_before(lambda p: uniform <= p, cumulative_pmf)

def len_before(f, iterable):
    return functools.reduce(lambda acc, x: acc if f(x) else acc+1, iterable, 0)

def log_optimal_strategy(prob, prices):
    x = cp.Variable(n)

    constraints = [
        x >= 0,
        np.ones(n) @ x == 1
    ]

    objective = cp.sum(cp.multiply(prob, cp.log(prices @ x)))

    problem = cp.Problem(cp.Maximize(objective), constraints)

    result = problem.solve()

    return (x.value, result)

if __name__ == '__main__':
    main()