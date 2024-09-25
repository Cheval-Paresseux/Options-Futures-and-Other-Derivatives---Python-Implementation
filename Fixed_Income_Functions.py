import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

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


# =========================================================================================================================================================================================================


class Bond:
    def __init__(
        self,
        face_value: float,
        coupon_rate: float,
        maturity: float,
        coupon_frequency: str,
        risk_free_rate: float,
        yield_to_maturity: float = None,
        market_price: float = None,
    ):
        self.face_value = face_value
        self.coupon_rate = coupon_rate
        self.maturity = maturity
        self.coupon_frequency = coupon_frequency
        self.rf = risk_free_rate
        self.yield_to_maturity = yield_to_maturity
        self.market_price = market_price

        # Deciding which rate we take into account
        if self.yield_to_maturity:
            self.discount_rate = self.yield_to_maturity
        else:
            self.discount_rate = self.rf

        # Changing type of frequency
        possible_frequencies = {
            "Monthly": 1 / 12,
            "Bi-monthly": 1 / 6,
            "Trimestrial": 1 / 4,
            "Semestral": 1 / 2,
            "Annual": 1,
            "2years": 2,
        }
        self.frequency = possible_frequencies[self.coupon_frequency]

    def coupon_value(self):
        """This function aims to get information about the coupons.

        Returns:
            coupon (float) -> the amount paid by a coupon
            coupons_pv (float) -> the present value of coupons
        """
        number_of_coupons = int(self.maturity / self.frequency)
        coupon = self.face_value * self.coupon_rate * self.frequency

        coupons_pv = sum(
            [
                coupon * np.exp(-self.discount_rate * t * self.frequency)
                for t in range(1, number_of_coupons + 1)
            ]
        )

        return number_of_coupons, coupon, coupons_pv

    def market_ytm(self):
        """Compute the market yield_to_maturity.

        Returns:
            yield_to_maturity (float) -> the market yield to maturity
        """
        if self.market_price:
            number_of_coupons, coupon, coupons_pv = self.coupon_value()

            def bond_price(r):
                coupons = sum(
                    [
                        coupon * np.exp(-r * t * self.frequency)
                        for t in range(1, number_of_coupons + 1)
                    ]
                )  # coupons present value
                face_value_discounted = self.face_value * np.exp(
                    -r * self.maturity
                )  # face_value present value
                return (
                    coupons + face_value_discounted - self.market_price
                )  # price = coupons + face_value discounted => we solve for =0

            # We use fsolve to solve the equation
            initial_guess = 0.05  # wanna take a bet ?
            ytm = fsolve(bond_price, initial_guess)
            yield_to_maturity = ytm[0]
        else:
            yield_to_maturity = None
            print("You did not give a market price")

        return yield_to_maturity

    def use_market_yield(self):
        """Let the user change the yield used to market rate."""
        if self.market_price:
            self.discount_rate = self.market_ytm()
        else:
            print("You did not give a market price")

    def bond_price(self):
        """Gets the bond_price

        Returns:
            bond_price (float) -> the discounted price of the bond
        """
        number_of_coupons, coupon, coupons_pv = self.coupon_value()
        face_value_pv = self.face_value * np.exp(-self.discount_rate * self.maturity)

        bond_price = coupons_pv + face_value_pv

        return bond_price

    def duration(self):
        """Compute the duration of the bond.

        Returns:
            duration (float) -> duration of the bond
        """
        number_of_coupons, coupon, coupons_pv = self.coupon_value()
        duration = sum(
            [
                t
                * self.frequency
                * coupon
                * np.exp(-self.discount_rate * t * self.frequency)
                for t in range(1, number_of_coupons + 1)
            ]
        ) + self.maturity * self.face_value * np.exp(
            -self.discount_rate * self.maturity
        )
        if self.market_price:
            duration = duration / self.market_price
        else:
            duration = duration / self.bond_price()

        return duration

    def convexity(self):
        """Compute the duration of the bond.

        Returns:
            duration (float) -> duration of the bond
        """
        number_of_coupons, coupon, coupons_pv = self.coupon_value()
        convexity = sum(
            [
                (t**2)
                * self.frequency
                * coupon
                * np.exp(-self.discount_rate * t * self.frequency)
                for t in range(1, number_of_coupons + 1)
            ]
        ) + (self.maturity**2) * self.face_value * np.exp(
            -self.discount_rate * self.maturity
        )
        if self.market_price:
            convexity = convexity / self.market_price
        else:
            convexity = convexity / self.bond_price()

        return convexity

    def yield_change(self, delta_yield: float, method: str):
        """Compute the chnage in price of the bond according to a change in yield.

        Args:
            delta_yield (float) -> change in the yield.
            method (str) -> "duration" for small changes / "convexity" for larger changes.

        Returns:
            change_return (float) -> change of the bond price expressed in a return
            bond_change (float) -> change of the bond price expressed in value
            new_bond_price (float) -> price of the bond after the change in yield
        """
        if method == "duration":
            bond_change = -self.bond_price() * self.duration() * delta_yield
            change_return = bond_change / self.bond_price()
        elif method == "convexity":
            bond_change = (
                -self.bond_price() * self.duration() * delta_yield
                + self.bond_price() * (1 / 2) * self.convexity() * delta_yield**2
            )
            change_return = bond_change / self.bond_price()
        else:
            print("Invalid method : it should be -duration- or -convexity-")

        new_bond_price = self.bond_price() + bond_change

        return change_return, bond_change, new_bond_price

    def exposition(self):
        """Used to visualize the exposition of the bond."""
        convex_list = []
        duration_list = []

        yield_changes = np.arange(
            -0.1, 0.1, 0.00001
        )  # Stocker les changements de rendement

        # Remplissage des listes
        for i in yield_changes:
            convex_change, bond_change, new_bond_price = self.yield_change(
                delta_yield=i, method="convexity"
            )
            duration_change, bond_change, new_bond_price = self.yield_change(
                delta_yield=i, method="duration"
            )
            convex_list.append(convex_change)
            duration_list.append(duration_change)

        # Plotting both series on the same graph
        plt.figure(figsize=(10, 6))
        plt.plot(yield_changes, convex_list, label="Convexity Change", color="blue")
        plt.plot(yield_changes, duration_list, label="Duration Change", color="red")
        plt.title("Exposition to a Change in Yield")
        plt.xlabel("Yield Change")
        plt.ylabel("Return Change")
        plt.legend()
        plt.grid(True)
        plt.show()


# =========================================================================================================================================================================================================
"""
Now we will go through some examples on how to use those functions. 
"""
# I. equivalence_rates
input_rate = 0.05
input_frequence = "Yearly"
desired_compounding_frequence = "Quarterly"

equiv_rate = equivalence_rates(
    input_rate, input_frequence, desired_compounding_frequence
)
print(f"Equivalent rate for quarterly compounding: {equiv_rate:.4f}")


# II. get_forward_price
spot_price = 100  # Spot price of the asset
risk_free_rate = 0.03  # 3% annual risk-free rate
maturity = 1  # 1 year

forward_price = get_forward_price(spot_price, risk_free_rate, maturity)
print(f"Forward price: {forward_price:.2f}")

# III. futures_hedging
position_to_hedge_size = 1000  # Size of the position to hedge
futures_contracts_size = 100  # Size of a single futures contract
correlation = 0.8  # Correlation between the asset and futures contract
subjacent_standard_deviation = 0.2  # Standard deviation of the asset's returns
futures_standard_deviation = (
    0.15  # Standard deviation of the futures contract's returns
)

hedge_ratio, number_of_contracts_needed = futures_hedging(
    position_to_hedge_size,
    futures_contracts_size,
    correlation,
    subjacent_standard_deviation,
    futures_standard_deviation,
)
print(f"Hedge ratio: {hedge_ratio:.2f}")
print(f"Number of futures contracts needed: {number_of_contracts_needed:.2f}")

# IV. class Bond
# Création d'une instance de Bond
bond = Bond(
    face_value=1000,
    coupon_rate=0.06,  # 6% coupon rate
    maturity=5,  # 5 years to maturity
    coupon_frequency="Annual",
    risk_free_rate=0.04,  # 4% risk-free rate
    market_price=980,  # Market price of the bond
)

# Calcul de la valeur du coupon
number_of_coupons, coupon, coupons_pv = bond.coupon_value()
print(f"Number of coupons: {number_of_coupons}")
print(f"Coupon value: {coupon:.2f}")
print(f"Present value of coupons: {coupons_pv:.2f}")

# Calcul du rendement à l'échéance
ytm = bond.market_ytm()
print(f"Yield to Maturity: {ytm:.4f}")

# Utilisation du rendement du marché
bond.use_market_yield()
print(f"Discount rate after using market yield: {bond.discount_rate:.4f}")

# Calcul du prix de l'obligation
bond_price = bond.bond_price()
print(f"Bond price: {bond_price:.2f}")

# Calcul de la duration
duration = bond.duration()
print(f"Duration: {duration:.2f}")

# Calcul de la convexité
convexity = bond.convexity()
print(f"Convexity: {convexity:.2f}")

# Calcul de la variation de prix en fonction du changement de rendement
delta_yield = 0.01  # 1% change in yield
change_return, bond_change, new_bond_price = bond.yield_change(
    delta_yield, method="convexity"
)
print(f"Change in bond price: {bond_change:.2f}")
print(f"Change in return: {change_return:.2f}")
print(f"New bond price: {new_bond_price:.2f}")

# Visualisation de l'exposition à un changement de rendement
bond.exposition()
