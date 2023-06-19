import numpy as np
import cvxpy as cp

### EE364a Homework 6 additional problems
# Exercise 2

n = 20
sigma = 0.2

def main():
    # M = generate_pure_random_matches()
    a_hidden =  np.random.random(n)
    M = generate_parametrized_random_matches(a_hidden)

    A = build_A(M)

    a = maximum_likelihood_abilities(A)

    M_pred = predict_matches(a)

    print('Prediction train error', np.sum(np.abs(M-M_pred))/(n * (n-1))/2)

    M_test = generate_parametrized_random_matches(a_hidden)
    print('Prediction test error', np.sum(np.abs(M_test-M_pred))/(n * (n-1))/2)

def generate_pure_random_matches():
    M = np.triu(np.sign(np.random.random((n, n)) - 0.5))

    return M - np.transpose(M)

def generate_parametrized_random_matches(a):
    M = np.zeros((n,n))

    for i in range(n):
        for j in range(i+1,n):
            M[i,j] = np.sign(a[i] - a[j] + np.random.normal(0, sigma))

    return M - np.transpose(M)

def predict_matches(a):
    M_pred = np.zeros((n,n))

    for i in range(n):
        for j in range(i+1,n):
            M_pred[i,j] = np.sign(a[i] - a[j])

    return M_pred - np.transpose(M_pred)
    
def build_A(M):
    A = np.zeros((n * (n-1) ,n))
    
    l = 0
    for i in range(n):
        for j in range(i+1, n):
            winner = M[i,j]
            A[l,i] = winner
            A[l,j] = - winner
            l += 1
    
    return A

def maximum_likelihood_abilities(A):
    a = cp.Variable(n)

    constraints = [ 
        a >= 0,
        a <= 1
    ]

    objective = cp.sum(cp.log_normcdf((A @ a) / sigma))

    problem = cp.Problem(cp.Maximize(objective), constraints)

    result = problem.solve()

    print('result', result)

    return a.value

if __name__ == '__main__':
    main()