import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

### EE364a Homework 4 additional problems
# Exercise 4 

def main():
    cutoff_frequencies = np.arange(np.pi/3 + np.pi/20, np.pi, np.pi/20)
    optimal_attenuations = [optimal_attenuation(7, cutoff_frequency) for cutoff_frequency in cutoff_frequencies]

    plt.plot(cutoff_frequencies, optimal_attenuations, '-')
    plt.show()

def optimal_attenuation(filter_size, cutoff_frequency):
    number_of_samples = filter_size * 10
    step_size = np.pi/number_of_samples

    k = np.arange(filter_size+1)

    passband = np.arange(0, np.pi/3 + step_size, step_size)
    stopband = np.arange(cutoff_frequency, np.pi + step_size, step_size)

    filter_magnitudes = cp.Variable(filter_size+1)

    constraints = [
        np.cos(np.outer(passband, k)) @ filter_magnitudes >= 0.89,
        np.cos(np.outer(passband, k)) @ filter_magnitudes <= 1.12,
    ]

    problem = cp.Problem(cp.Minimize(cp.norm_inf(np.cos(np.outer(stopband, k)) @ filter_magnitudes)), constraints)
    
    return problem.solve()

if __name__ == '__main__':
    main()