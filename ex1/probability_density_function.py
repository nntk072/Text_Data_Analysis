import numpy as np
import math


def gaussian(x, mu, sigma):
    """
    :param x: evaluation location
    :param mu: mean vector
    :param sigma: covariance matrix
    :return: probability density function of a multivariate Gaussian distribution
    """

    det = np.linalg.det(sigma)
    inv = np.linalg.inv(sigma)

    exponent = -0.5 * np.dot(np.dot(np.transpose(x - mu), inv), (x - mu))
    coefficient = 1 / (math.sqrt((2 * math.pi) ** len(x) * det))

    return coefficient * math.exp(exponent)


if __name__ == "__main__":
    mu = np.array([1, 3, 5])
    sigma = np.array([[4, 2, 1], [2, 5, 2], [1, 2, 3]])
    x1 = np.array([2, 2, 2])
    x2 = np.array([1, 4, 3])
    x3 = np.array([1, 1, 5])

    print("Gaussian distribution:")
    print(f"mu = {mu}")
    print(f"sigma = \n{sigma}")
    print("Probability density function value:")
    print(f"case 1: {x1}: {gaussian(x1, mu, sigma)}")
    print(f"case 2: {x2}: {gaussian(x2, mu, sigma)}")
    print(f"case 3: {x3}: {gaussian(x3, mu, sigma)}")
