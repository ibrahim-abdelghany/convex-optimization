import numpy as np
import cvxpy as cp

### EE364a Homework 6 additional problems
# Exercise 4

n = 5 # dimension of parameters to be estimated
p = 20 # number of available types of measurements
m = 30 # total number of measurements to be carried out

def main():
    V = np.random.normal(size=(n,p)) # columns are vi, the possible measurement vectors
    
    relaxed_optimal, lambdas = relaxed_a_optimal_experiment_design(V)

    print('Relaxed optimal', relaxed_optimal)
    print('Relaxed lambdas', lambdas)

    m_is = np.round(lambdas * m)

    integer_optimal = np.trace(np.linalg.inv(V @ np.diag(m_is) @ np.transpose(V)))

    print('Integer error', integer_optimal, 'deviation', integer_optimal - relaxed_optimal)
    print('Integer weights', m_is, np.sum(m_is))

def relaxed_a_optimal_experiment_design(V):
    lambdas = cp.Variable(p)

    inner = V @ cp.diag(lambdas) @ cp.transpose(V)

    # cp.tr_inv has an implementation issue, this workaround allows us to do tr(inner^-1)
    objective = cp.matrix_frac(np.eye(n), inner)/m

    constraints = [
        lambdas >= 0,
        cp.sum(lambdas) == 1
    ]
    
    problem = cp.Problem(cp.Minimize(objective), constraints)

    result = problem.solve()

    return (result, lambdas.value)

if __name__ == '__main__':
    main()