import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
import scipy.stats as stats
import seaborn as sns


# Load cleaned data
data = pd.read_csv("Strategic_Portfolio_Cleaned_Data.csv", parse_dates=["Date"])

data.set_index("Date", inplace=True)

print(data.head())

log_returns = np.log(data / data.shift(1))
log_returns.dropna(inplace=True)

print("\nLog Returns:")
print(log_returns.head())

mean_returns = log_returns.mean() * 252
cov_matrix = log_returns.cov() * 252

num_assets = len(data.columns)
weights = np.ones(num_assets) / num_assets

portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
portfolio_volatility = np.sqrt(portfolio_variance)

print("\nAnnual Portfolio Volatility:", portfolio_volatility)

num_simulations = 10000
T = 252

simulated_ending_values = []

for _ in range(num_simulations):
    simulated_returns = np.random.multivariate_normal(
        log_returns.mean(),
        log_returns.cov(),
        T
    )

    portfolio_returns = simulated_returns.dot(weights)
    portfolio_growth = np.exp(np.cumsum(portfolio_returns))

    simulated_ending_values.append(portfolio_growth[-1])

simulated_ending_values = np.array(simulated_ending_values)

VaR_95 = np.percentile(simulated_ending_values, 5)

print("\n95% Value at Risk (VaR):", VaR_95)
VaR_95 = np.percentile(simulated_ending_values, 5)

VaR_loss = 1 - VaR_95

print("\n95% Value at Risk (Growth Level):", VaR_95)
print("95% Potential Loss:", VaR_loss * 100, "%")


print("\nDistribution Statistics:")
print("Mean:", np.mean(simulated_ending_values))
print("Std Dev:", np.std(simulated_ending_values))
print("Skewness:", skew(simulated_ending_values))
print("Kurtosis:", kurtosis(simulated_ending_values))

plt.hist(simulated_ending_values, bins=50)
plt.axvline(np.mean(simulated_ending_values), linestyle='dashed')
plt.axvline(VaR_95, linestyle='dashed')
plt.title("Monte Carlo Portfolio Distribution")
plt.xlabel("Portfolio Growth")
plt.ylabel("Frequency")
plt.show()

stats.probplot(simulated_ending_values, dist="norm", plot=plt)
plt.title("QQ Plot - Normality Check")
plt.show()


sns.histplot(simulated_ending_values, bins=50, kde=True)
plt.title("Monte Carlo Distribution with KDE")
plt.show()


plt.figure(figsize=(10,6))

sns.histplot(simulated_ending_values, bins=50, kde=True)

plt.axvline(np.mean(simulated_ending_values), linestyle='dashed', linewidth=2, label="Mean")
plt.axvline(VaR_95, linestyle='dashed', linewidth=2, label="95% VaR")

plt.title("Monte Carlo Simulation - Portfolio Distribution")
plt.xlabel("Portfolio Growth After 1 Year")
plt.ylabel("Frequency")
plt.legend()

plt.show()