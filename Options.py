import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

######################################################################################################################################################################################
######################################################################################################################################################################################
######################################################################################################################################################################################


class VanillaOption:
    """
    This class aims to gather different pricing methods for Options.
    For the moment it covers only European options.
    If prices history given, it should be daily prices.
    """

    def __init__(
        self,
        type: str,
        spot: float,
        strike: float,
        maturity: float,
        risk_free_rate: float = None,
        subjacent_volatility: float = None,
        implied_volatility: float = None,
        subjacent_prices: pd.Series = None,
        options_prices: pd.Series = None,
    ):
        self.type = type.lower()
        self.spot = spot
        self.K = strike
        self.maturity = maturity
        self.rf = risk_free_rate if risk_free_rate is not None else 0.0
        self.sub_vol = subjacent_volatility
        self.imp_vol = implied_volatility
        self.sub_history = subjacent_prices
        self.opt_history = options_prices
        self.vol = 0

        if not (
            subjacent_volatility
            or implied_volatility
            or subjacent_prices
            or options_prices
        ):
            print("Warning: No way to compute volatility was specified.")

    # ---------------------------------------------------------------------------------------------------------------------#

    def payoff(
        self, position: str, spot_price: float, option_market_price: float = None
    ):
        """
        Compute the payoff of the option for a given spot price.
        """
        option_cost = option_market_price if option_market_price is not None else 0.0

        if self.type == "call":
            payoff = max(0, spot_price - self.K) - option_cost
        elif self.type == "put":
            payoff = max(0, self.K - spot_price) - option_cost
        else:
            raise ValueError(
                f"Unsupported option type '{self.type}'. Must be 'call' or 'put'."
            )

        if position.lower() == "short":
            payoff = -payoff
        elif position.lower() != "long":
            raise ValueError(
                f"Unsupported position '{position}'. Must be 'long' or 'short'."
            )

        return payoff

    # ---------------------------------------------------------------------------------------------------------------------#

    def payoff_graph(self, position: str, option_market_price: float = None):
        """
        Generate a payoff graph for the option based on the specified position (long/short).
        """
        # =========== I. Determine the range of spot prices to evaluate
        price_step = self.spot / 1000  # Small incremental step for spot prices
        min_price = self.spot - 500 * price_step  # Start 50% below the spot price

        # =========== II. Loop through the spot price range and compute payoff
        spot_prices = []
        option_payoffs = []
        for i in range(1001):
            current_spot_price = min_price + price_step * i
            payoff = self.payoff(
                position=position,
                spot_price=current_spot_price,
                option_market_price=option_market_price,
            )
            spot_prices.append(current_spot_price)
            option_payoffs.append(payoff)

        # =========== III. Plotting the payoff graph
        plt.figure(figsize=(10, 6))
        plt.plot(
            spot_prices, option_payoffs, label=f"{self.type.capitalize()} Option Payoff"
        )
        plt.axhline(0, color="black", lw=1)  # Horizontal axis
        plt.axvline(
            self.K, color="red", linestyle="--", label="Strike Price"
        )  # Strike price line
        plt.title(
            f"{self.type.capitalize()} Option Payoff for {position.capitalize()} Position"
        )
        plt.xlabel("Spot Price")
        plt.ylabel("Payoff")
        plt.legend()
        plt.grid(True)
        plt.show()

    # ---------------------------------------------------------------------------------------------------------------------#

    def set_volatility(self, choice: str = None, custom_vol: float = None):
        """
        This methods fixes the volaitility used for pricing models.
        Args:
            choice (str): possible arguments are ['subjacent', 'implied']
            custom_vol (float): possibility to input a custom volatility
        """
        if custom_vol is not None:
            self.vol = custom_vol
            return

        if choice is None:
            raise ValueError(
                "Specify a method to set volatility or provide custom_vol."
            )

        choice = choice.lower()
        if choice == "subjacent":
            if self.sub_vol:
                self.vol = self.sub_vol
            elif self.sub_history is not None:
                self.vol = self.sub_history.std() * np.sqrt(252)
            else:
                raise ValueError(
                    "Subjacent price history not provided for subjacent volatility calculation."
                )

        elif choice == "implied":
            if self.imp_vol:
                self.vol = self.imp_vol
            elif self.opt_history is not None:
                self.vol = self.opt_history.std() * np.sqrt(252)
            else:
                raise ValueError(
                    "Option price history not provided for implied volatility calculation."
                )
        else:
            raise ValueError(f"Unknown choice '{choice}' for volatility calculation.")

    # ---------------------------------------------------------------------------------------------------------------------#

    def binomial_tree_pricing(
        self,
        number_of_branch: int,
        option_style: str = "european",  # New argument to select option style
        custom_up: float = None,
        custom_down: float = None,
    ):
        """
        Pricing model of binomial tree with support for both American and European options.
        Args:
            number_of_branch (int): the more branches, the more precise, but higher computational cost.
            option_style (str): 'european' or 'american' to specify the type of option.
            custom_up & custom_down (float): possibility to choose custom parameters (Warning : they should match the number of branch)
        Returns:
            option_value (float): the computed price of the option
        """
        # =========== I. Check if we have all necessary parameters to compute.
        if (custom_down and not custom_up) or (custom_up and not custom_down):
            raise ValueError("Need to specify both custom_up and custom_down.")

        if self.vol == 0 and not (custom_up and custom_down):
            raise ValueError(
                "Need to set volatility or specify custom up and down parameters."
            )

        # =========== II. Set up the parameters (dt, u, d, p)
        dt = self.maturity / number_of_branch

        # Set up and down factors
        if custom_up is not None and custom_down is not None:
            u = 1 + custom_up
            d = 1 - custom_down
        else:
            u = np.exp(self.vol * np.sqrt(dt))
            d = np.exp(-self.vol * np.sqrt(dt))

        # Risk-neutral probability
        p = (np.exp(self.rf * dt) - d) / (u - d)

        if p < 0 or p > 1:
            raise ValueError(
                "Invalid probability: check volatility or input parameters."
            )

        # =========== III. Compute stock prices at each node
        tree = np.zeros((number_of_branch + 1, number_of_branch + 1))
        tree[0, 0] = self.spot
        for i in range(1, number_of_branch + 1):
            tree[i, 0] = tree[i - 1, 0] * u
            for j in range(1, i + 1):
                tree[i, j] = tree[i - 1, j - 1] * d

        # =========== IV. Compute option value at maturity
        option_values = np.zeros((number_of_branch + 1))
        for j in range(number_of_branch + 1):
            if self.type == "call":
                option_values[j] = max(0, tree[number_of_branch, j] - self.K)
            elif self.type == "put":
                option_values[j] = max(0, self.K - tree[number_of_branch, j])

        # =========== V. Backward induction
        for i in range(number_of_branch - 1, -1, -1):
            for j in range(i + 1):
                # Calculate the value by holding the option
                option_values[j] = (
                    p * option_values[j] + (1 - p) * option_values[j + 1]
                ) * np.exp(-self.rf * dt)

                # If American option, check for early exercise
                if option_style == "american":
                    if self.type == "call":
                        exercise_value = max(0, tree[i, j] - self.K)
                    elif self.type == "put":
                        exercise_value = max(0, self.K - tree[i, j])
                    option_values[j] = max(option_values[j], exercise_value)

        return option_values[0]

    # ---------------------------------------------------------------------------------------------------------------------#


######################################################################################################################################################################################
######################################################################################################################################################################################
######################################################################################################################################################################################

eurocall = VanillaOption(
    type="put",
    spot=100,
    strike=105,  # Slightly in-the-money
    maturity=2,  # Longer maturity
    risk_free_rate=0.05,
    subjacent_volatility=0.30,  # High volatility
)
eurocall.set_volatility(choice="subjacent")

european_price = eurocall.binomial_tree_pricing(
    number_of_branch=100, option_style="european"
)
american_price = eurocall.binomial_tree_pricing(
    number_of_branch=100, option_style="american"
)

print(f"European : {european_price}", f"American : {american_price}")
