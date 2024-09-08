import numpy as np

# =========================================================================================================================================================================================================


def equivalence_rates(
    input_rate: float, input_frequence: str, desired_compounding_frequence: str
) -> float:
    """
    This function computes the equivalent rate for a given compounding frequency.

    Args:
        input_rate (float): The annual interest rate for which we want an equivalent rate.
        input_frequence (str): The compounding frequency of the input rate, expressed in terms such as
                            'Yearly', 'Semi-annual', 'Quarterly', 'Monthly', or 'Continuous'.
        desired_compounding_frequence (str): The desired compounding frequency to calculate the equivalent rate, expressed in terms such as
                            'Yearly', 'Semi-annual', 'Quarterly', 'Monthly', or 'Continuous'.


    Returns:
        equivalent_rate (float): The equivalent rate for the desired compounding frequency.
    """

    # Define the number of compounding periods per year for each frequency
    frequencies = {
        "Yearly": 1,
        "Semi-annual": 2,
        "Quarterly": 4,
        "Monthly": 12,
        "Continuous": "Continuous",
    }

    # Ensure valid frequencies are provided
    if (
        input_frequence not in frequencies
        or desired_compounding_frequence not in frequencies
    ):
        raise ValueError(
            "Invalid compounding frequency. Choose from 'Yearly', 'Semi-annual', 'Quarterly', 'Monthly', or 'Continuous'."
        )

    # Handle the case of continuous compounding
    if frequencies[input_frequence] == "Continuous":
        input_effective_rate = np.exp(input_rate) - 1
    else:
        input_effective_rate = (
            1 + input_rate / frequencies[input_frequence]
        ) ** frequencies[input_frequence] - 1

    if frequencies[desired_compounding_frequence] == "Continuous":
        equivalent_rate = np.log(1 + input_effective_rate)
    else:
        equivalent_rate = frequencies[desired_compounding_frequence] * (
            (1 + input_effective_rate)
            ** (1 / frequencies[desired_compounding_frequence])
            - 1
        )

    return equivalent_rate


# Example usage:
input_rate = 0.05  # 5% annual interest rate
input_frequence = "Yearly"
desired_compounding_frequence = "Quarterly"

equivalent_rate = equivalence_rates(
    input_rate, input_frequence, desired_compounding_frequence
)
print(
    f"The equivalent rate for {desired_compounding_frequence} compounding is: {equivalent_rate:.6f}"
)

# =========================================================================================================================================================================================================


def bond_pricer(
    face_value: float,
    risk_free_rate: float,
    coupon_rate: float,
    coupon_frequence: float,
    maturity: float,
):
    """
    This function computes the price of bonds given the parameters.
    Args:
        face_value (float): The face value of the bond.
        risk_free_rate (float): The annual risk-free rate for the same maturity as the bond.
        coupon_rate (float): The annual coupon rate.
        coupon_frequence (float): The frequency of coupon payments (semi-annual = 0.5, quarterly = 0.25, etc.).
        maturity (float): The maturity of the bond in years.

    Returns:
        bond_price (float): The bond value at time t=0.
    """
    # Present value of face value
    present_value_of_face = face_value * np.exp(-risk_free_rate * maturity)

    # Present value of coupons
    present_value_of_coupons = 0
    number_of_coupons = int(maturity / coupon_frequence)
    if coupon_rate > 0:
        for i in range(1, number_of_coupons + 1):
            coupon_pv = (
                face_value
                * coupon_rate
                * coupon_frequence
                * np.exp(-risk_free_rate * coupon_frequence * i)
            )
            present_value_of_coupons += coupon_pv

    # Bond price
    bond_price = present_value_of_face + present_value_of_coupons

    return bond_price


# =========================================================================================================================================================================================================


def get_forward_price(
    spot_price: float, risk_free_rate: float, maturity: float
) -> float:
    """
    This function computes the forward price of an asset.

    Args:
        spot_price (float): The spot price of the underlying asset.
        risk_free_rate (float): The annual risk-free rate for the given maturity.
        maturity (float): The time to maturity, expressed in years.

    Returns:
        forward_price (float): The calculated forward price.
    """
    forward_price = spot_price * np.exp(risk_free_rate * maturity)

    return forward_price


# =========================================================================================================================================================================================================


def futures_hedging(
    position_to_hedge_size: float,
    futures_contracts_size: float,
    correlation: float,
    subjacent_standard_deviation: float,
    futures_standard_deviation: float,
):
    """
    This function computes the hedge ratio and the number of futures contracts needed to hedge a position.

    Args:
        position_to_hedge_size (float): The size of the position to hedge (e.g., number of units of the asset).
        futures_contracts_size (float): The size of a single futures contract (e.g., number of units per contract).
        correlation (float): The correlation between the asset to hedge and the futures contract.
        subjacent_standard_deviation (float): The standard deviation of the underlying asset's returns.
        futures_standard_deviation (float): The standard deviation of the futures contract's returns.

    Returns:
        hedge_ratio (float): The optimal hedge ratio (also known as the hedge effectiveness).
        number_of_contracts_needed (float): The number of futures contracts needed to hedge the position.
    """
    # Calculate hedge ratio (optimal hedge ratio)
    hedge_ratio = (
        correlation * subjacent_standard_deviation / futures_standard_deviation
    )

    # Calculate the number of futures contracts needed for hedging
    number_of_contracts_needed = (
        hedge_ratio * position_to_hedge_size / futures_contracts_size
    )

    return hedge_ratio, number_of_contracts_needed
