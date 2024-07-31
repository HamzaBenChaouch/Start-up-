import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the cleaned dataset
startup_data_cleaned = pd.read_csv('data/startup_data_cleaned.csv')

# Distribution of Success Labels
plt.figure(figsize=(8, 6))
sns.countplot(x='labels', data=startup_data_cleaned)
plt.title('Distribution of Success Labels')
plt.xlabel('Success (1) / Failure (0)')
plt.ylabel('Count')
plt.savefig('plots/success_labels_distribution.png')
plt.show()

# Improved Histogram for Total Funding Amounts
plt.figure(figsize=(10, 6))
sns.histplot(startup_data_cleaned['funding_total_usd'], bins=100, kde=True)
plt.xscale('log')  # Use a logarithmic scale for the x-axis
plt.title('Distribution of Total Funding Amounts (USD)')
plt.xlabel('Total Funding (USD)')
plt.ylabel('Count')
plt.savefig('plots/total_funding_distribution.png')
plt.show()

# Correlation heatmap
numeric_cols = startup_data_cleaned.select_dtypes(include=['float64', 'int64']).columns
plt.figure(figsize=(14, 12))
sns.heatmap(startup_data_cleaned[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap of Numerical Features')
plt.savefig('plots/correlation_heatmap.png')
plt.show()

# Funding Rounds vs Success Labels
plt.figure(figsize=(10, 6))
sns.boxplot(x='labels', y='funding_rounds', data=startup_data_cleaned)
plt.title('Funding Rounds vs Success Labels')
plt.xlabel('Success (1) / Failure (0)')
plt.ylabel('Number of Funding Rounds')
plt.savefig('plots/funding_rounds_vs_success_labels.png')
plt.show()
