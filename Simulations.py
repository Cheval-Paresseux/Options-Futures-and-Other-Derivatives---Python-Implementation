import numpy as np

######################################################################################################################################################################################
######################################################################################################################################################################################
######################################################################################################################################################################################


def GBM_simulation(
    X0: float,
    expected_return: float,
    volatility: float,
    total_time: int,
    number_of_observations: int,
):
    """
    Simulate a geometric Brownian motion (GBM) to model stock prices.

    Parameters:
        X0 (float): Initial stock price
        expected_return (float): Expected return (drift, mu)
        volatility (float): Volatility of the stock (sigma)
        total_time (float): Total time (in years, or other unit of time)
        number_of_obersvations (int): Number of time steps

    Returns:
        prices (np.array) Simulated stock prices over time
    """
    # =========== I. Set up parameters
    dt = total_time / number_of_observations
    prices = np.zeros(number_of_observations)
    prices[0] = X0

    # =========== II. Simulate the GBM process
    random_shocks = np.random.normal(0, 1, number_of_observations)
    for t in range(1, number_of_observations):
        prices[t] = prices[t - 1] * np.exp(
            (expected_return - 0.5 * volatility**2) * dt
            + volatility * np.sqrt(dt) * random_shocks[t]
        )

    return prices


######################################################################################################################################################################################


def GBM_monte_carlo_simulation(
    X0: float,
    expected_return: float,
    volatility: float,
    total_time: float,
    number_of_observations: int,
    number_of_simulations: int,
):
    """
    Perform a Monte Carlo simulation of stock prices using Geometric Brownian Motion (GBM).

    Parameters:
        X0 (float): Initial stock price
        expected_return (float): Expected return (drift, mu)
        volatility (float): Volatility of the stock (sigma)
        total_time (float): Total time (in years, or other unit of time)
        number_of_observations (int): Number of time steps per simulation
        number_of_simulations (int): Number of Monte Carlo simulations (paths)

    Returns:
       final_prices (np.array): Simulated final prices from each path
       all_simulations (np.array): Matrix of all simulated price paths (M simulations by N time steps)
    """
    # =========== I. Generate simulations
    all_simulations = np.zeros((number_of_simulations, number_of_observations))

    for i in range(number_of_simulations):
        all_simulations[i, :] = GBM_simulation(
            X0=X0,
            expected_return=expected_return,
            volatility=volatility,
            total_time=total_time,
            number_of_observations=number_of_observations,
        )

    # =========== II. Get final prices
    final_prices = all_simulations[:, -1]

    return final_prices, all_simulations
