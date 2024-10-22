import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Set the page configuration
st.set_page_config(
    page_title="Portfolio Optimizer",  # Change this to your desired title
    page_icon="🪙",  # Optional: Change the icon (can be a string, e.g., emoji or a URL to an image)
)

# Streamlit configuration
st.title("Stock Portfolio Risk and Returns Dashboard")

# User input: Stock tickers
tickers = st.text_input(
    "Enter stock tickers (separated by commas)", "COIN,VOO,LMND,ACHR"
)
tickers = tickers.split(",")

# User input: Portfolio weights
weights_input = st.text_input(
    "Enter portfolio weights (comma-separated, must sum to 1.0) (optional) or leave blank for recommended weights"
)
if weights_input.strip() == "":
    weights = None  # User did not provide weights
else:
    weights = np.array([float(w) for w in weights_input.split(",")])

# Date range slider for historical data
start_date = st.date_input("Select start date", value=pd.to_datetime("2014-01-01"))
end_date = st.date_input("Select end date", value=pd.to_datetime("2024-01-01"))
# Download stock data using yfinance
data = yf.download(tickers, start=start_date, end=end_date)["Adj Close"]

# Check if the data is empty
if data.empty:
    st.error("No data retrieved. Check the ticker symbols or date range.")
else:
    # Display first 25 rows of the data
    st.subheader("Sample of Downloaded Data (Top 10)")  # need to change order
    st.dataframe(data.tail(10))

    # Calculate daily returns
    returns = data.pct_change().dropna()

    # If weights are not provided, calculate optimal weights based on Sharpe ratio
    if weights is None:
        # Calculate expected returns and covariance matrix
        mean_returns = returns.mean() * 252  # Annualized mean returns
        cov_matrix = returns.cov() * 252  # Annualized covariance matrix

        num_assets = len(tickers)

        # Function to calculate portfolio Sharpe ratio
        def calculate_sharpe(weights):
            portfolio_return = np.dot(weights, mean_returns)
            portfolio_volatility = np.sqrt(
                np.dot(weights.T, np.dot(cov_matrix, weights))
            )
            return (
                portfolio_return - 0.02
            ) / portfolio_volatility  # Assuming risk-free rate is 2%

        # Optimize weights for maximum Sharpe ratio
        from scipy.optimize import minimize

        initial_weights = np.array(
            num_assets * [1.0 / num_assets]
        )  # Start with equal weights

        constraints = {
            "type": "eq",
            "fun": lambda x: np.sum(x) - 1,
        }  # Weights must sum to 1
        bounds = tuple(
            (0, 1) for _ in range(num_assets)
        )  # Weights must be between 0 and 1

        optimal_weights = minimize(
            lambda x: -calculate_sharpe(x),
            initial_weights,
            bounds=bounds,
            constraints=constraints,
        )
        weights = optimal_weights.x  # Assign the optimal weights
        # Convert to percentages and round for better readability
        recommended_weights_percent = np.round(weights * 100, 2)

    # Validate if weights sum to 1
    if np.sum(weights) != 1:
        st.error("Portfolio weights must sum to 1.0")
    else:
        # User input: Initial investment
        initial_value = st.number_input(
            "Enter initial investment amount", value=25000, min_value=1
        )

        # User input: Number of years for projection
        n_years = st.number_input(
            "Enter number of years for projection", value=5, min_value=1
        )

        # User input: Number of Monte Carlo simulations
        num_simulations = st.number_input(
            "Number of Monte Carlo simulations", value=1000, min_value=1
        )

        # Create a DataFrame to hold the ticker symbols and their corresponding weights
        weights_df = pd.DataFrame(
            {
                "Ticker": [ticker.strip() for ticker in tickers],
                "Recommended Weight (%)": recommended_weights_percent,
            }
        )

        # Display recommended weights in a table format
        st.subheader("Recommended Portfolio Weights")
        st.table(weights_df)

        # Calculate portfolio returns
        portfolio_returns = returns.dot(weights)

        # Check if portfolio returns are empty
        if portfolio_returns.empty:
            st.error(
                "Portfolio returns could not be calculated. Please adjust your inputs."
            )
        else:
            # Calculate expected annual return
            mean_daily_return = portfolio_returns.mean()
            annual_return = mean_daily_return * 252

            # Calculate portfolio volatility (annualized)
            portfolio_volatility = np.sqrt(
                np.dot(weights.T, np.dot(returns.cov() * 252, weights))
            )

            # Calculate Sharpe ratio
            risk_free_rate = 0.04
            sharpe_ratio = (annual_return - risk_free_rate) / portfolio_volatility

            # Calculate Value at Risk (VaR)
            confidence_level = 0.05
            VaR = np.percentile(portfolio_returns, confidence_level * 100)

            # Portfolio projection after n_years
            projected_value = initial_value * (1 + annual_return) ** n_years

            # Display metrics
            st.subheader("Portfolio Performance Metrics")
            st.write(f"Expected Annual Return: {annual_return:.2%}")
            st.write(f"Annualized Volatility: {portfolio_volatility:.2%}")
            st.write(f"Sharpe Ratio: {sharpe_ratio:.2f}")
            st.write(f"Value at Risk (VaR) at 95% confidence: {VaR:.2%}")
            st.write(
                f"Projected Portfolio Value from the expected annual return after {n_years} years: ${projected_value:.2f}"
            )

            # Plot cumulative returns
            st.subheader(
                "Portfolio Cumulative Returns Till Date Based on Historical Data"
            )
            cumulative_returns = (1 + portfolio_returns).cumprod()
            plt.figure(figsize=(10, 6))
            cumulative_returns.plot(title="Portfolio Cumulative Returns")
            plt.ylabel("Cumulative Returns")
            st.pyplot(plt)

            # Monte Carlo simulation for future portfolio value
            simulations = np.zeros((num_simulations, n_years))
            for i in range(num_simulations):
                for t in range(n_years):
                    random_return = np.random.normal(
                        annual_return / 252, portfolio_volatility / np.sqrt(252)
                    )
                    simulations[i, t] = initial_value * (1 + random_return) ** (t + 1)

            # Plot simulation results
            st.subheader("Monte Carlo Simulations of Future Portfolio Value")
            plt.figure(figsize=(10, 6))
            plt.plot(simulations.T, color="blue", alpha=0.1)
            plt.title("Monte Carlo Simulations of Portfolio Value")
            plt.xlabel("Years")
            plt.ylabel("Portfolio Value")
            st.pyplot(plt)

            # Plot histogram of final portfolio values
            st.subheader("Distribution of Final Portfolio Values in the Future")
            final_values = simulations[:, -1]
            plt.figure(figsize=(10, 6))
            plt.hist(final_values, bins=30, color="orange", alpha=0.7)
            plt.title("Distribution of Final Portfolio Values After Simulations")
            plt.xlabel("Final Portfolio Value")
            plt.ylabel("Frequency")
            st.pyplot(plt)
