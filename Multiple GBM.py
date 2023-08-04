import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.linalg import cholesky

"""
for single option only
"""

def black_scholes(S, K, r, sigma, T):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    Nd1 = norm.cdf(d1)
    Nd2 = norm.cdf(d2)
    option_price = S*Nd1 - K*np.exp(-r*T)*Nd2
    return option_price

# Parameters
# input risk free rate under risk neutral pricing or drift under real world
mu = 0.0533
n = 1000
T = 1
M = 1000
initial_stock_prices = [4536.34, 4536.34, 4536.34]
sigma = 0.1817

#Change the floor rate
floor_rate = [0,0,0]
floor = initial_stock_prices * (np.ones_like(floor_rate) + floor_rate)
# Compute the Black-Scholes option price for each initial stock price
analytical_solution = black_scholes(initial_stock_prices[0], floor[0], mu, sigma, T)
print(analytical_solution)

# Correlation coefficients
corr12, corr13, corr23 = 0.7, 0.8, 0.9
correlation_matrix = np.array([[1, corr12, corr13], 
                               [corr12, 1, corr23], 
                               [corr13, corr23, 1]])

# Cholesky decomposition of the correlation matrix
L = cholesky(correlation_matrix, lower=True)

# Empty 3-d array to store the simulated stock price for each time step, for each path, and for each stock
St = np.zeros((3, n+1, M))
# Initialize the stock prices
St[:, 0, :] = np.array([initial_stock_prices] * M)


# Time step
dt = T / n

# Simulate the stock prices
for t in range(1, n+1):
    z = np.random.normal(0, 1, size=(3, M))
    epsilon = np.dot(L, z)
    St[:, t, :] = St[:, t-1, :] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * epsilon)

# Calculate the payoffs and fair prices
payoffs = np.maximum(0, St[:, -1, :] - np.array(floor)[:, np.newaxis])
expected_payoffs = np.mean(payoffs, axis=1)
fair_prices = expected_payoffs * np.exp(-mu * T)

# Print the fair prices for each dimension
print("Fair price of the options:", fair_prices)

# Plot the paths for each dimension separately
for i in range(3):
    plt.figure(i)
    plt.plot(St[i, :, :20])
    plt.xlabel("Years $(t)$")
    plt.ylabel("Stock Price $(S_t)$")
    plt.title(
        "Realizations of Geometric Brownian Motion\n $dS_t = \mu S_t dt + \sigma S_t dW_t$\n $S_0 = {0}, \mu = {1}, \sigma = {2}$".format(initial_stock_prices[i], mu, sigma)
    )
    plt.show()
