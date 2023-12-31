# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 15:45:37 2023

@author: Archana
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 19:45:29 2023

@author: Archana
"""

import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# Load and merge datasets
selected_columns = ['Industry', 'Sector']
df = pd.read_csv(r'D:\Users\Archana\Downloads\FinBert_Sentiments\merged_with_returns_earnings_date.csv')
sectors = pd.read_csv(r'D:\Users\Archana\Downloads\Sectors\all_sectors.csv')
merged_df = df.merge(sectors[selected_columns], on=['Industry'])
merged_df.sort_values(by=['Year', 'Quarter'], ascending=[True, True], inplace=True)

# Feature engineering based on sentiment and market cap
merged_df['MomentumXsentiment'] = merged_df['12_month_momentum'] * merged_df['Doc Sentiment Score']
merged_df['PositiveXsentiment'] = merged_df['Positive Sentiment Score'] * merged_df['Doc Sentiment Score']
merged_df = merged_df[merged_df['Market Cap'] > 3000000000]  # only considering mid-cap and large-cap stocks

# Drop unnecessary column 
if 'Unnamed: 0' in merged_df.columns:
    merged_df.drop(['Unnamed: 0'], axis=1, inplace=True)

# Sorting the DataFrame
merged_df.sort_values(by=['Year', 'Quarter'], ascending=[True, True], inplace=True)

# Plotting a scatter plot
sns.scatterplot(data=merged_df, x='Doc Sentiment Score', y='Relative_Returns_Industry_Average_6_month')
plt.xlabel('Doc Sentiment Score')
plt.ylabel('Relative Returns 1-month Avg')
plt.title('Scatter Plot of Sentiment Score vs. Relative Returns')
plt.ylim(-5, 10)
plt.show()

# Calculate and visualize IQR (Interquartile Range)
Q1 = merged_df['Relative_Returns_Industry_Average_1_month'].quantile(0.25)
Q3 = merged_df['Relative_Returns_Industry_Average_1_month'].quantile(0.75)
IQR = Q3 - Q1

# Use winsorizing to handle outliers
lower_bound = max(Q1 - 1.5 * IQR, merged_df['Relative_Returns_Industry_Average_1_month'].min())
upper_bound = Q3 + 1.5 * IQR
merged_df['Relative_Returns_Industry_Average_1_month'] = winsorize(
    merged_df['Relative_Returns_Industry_Average_1_month'],
    limits=(0.05, 0.05)
)

# Create bins and bucketize returns
bin_edges = merged_df['Relative_Returns_Industry_Average_1_month'].quantile([0, 0.2, 0.4, 0.6, 0.8, 1]).tolist()
bin_labels = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
merged_df['Return_Buckets'] = pd.cut(merged_df['Relative_Returns_Industry_Average_1_month'], bins=bin_edges,
                                     labels=bin_labels)

# Map labels to numerical values
ordinal_mapping = {'Very Low': 1, 'Low': 2, 'Medium': 3, 'High': 4, 'Very High': 5}
merged_df['Return_Buckets'] = merged_df['Return_Buckets'].map(ordinal_mapping)

# Drop rows with missing values
merged_df.dropna(subset=['Negative Sentiment Count', 'Positive Sentiment Count', 'Neutral Sentiment Count',
                          'Doc Sentiment Score', 'MomentumXsentiment', '1_month_momentum', '3_month_momentum',
                          '12_month_momentum', 'Return_Buckets', 'PositiveXsentiment'], inplace=True)

"""
Backtesting strategy:
Selling the top 20% of stocks and buying the bottom 20% based on sentiment score.
Data includes only mid-cap and large-cap stocks.

"""

merged_df['Date'] = pd.to_datetime(merged_df['Date'])
import pandas as pd
import yfinance as yf
from datetime import datetime
from dateutil.relativedelta import relativedelta

# Set start and end years, and investment per stock
start_year = 2019
end_year = 2023
investment_per_stock = 100

# Initialize a list to store sector returns
sector_returns = []

# Loop through each year and month
for year in range(start_year, end_year + 1):
    for month in range(1, 13):
        start_date = datetime(year, month, 1)
        end_date = start_date + relativedelta(months=12)
        max_start_date = start_date - relativedelta(months=2)

        # Filter the DataFrame based on date range
        df = merged_df[(merged_df['Date'] >= max_start_date) & (merged_df['Date'] <= start_date)]

        # Initialize a dictionary to store sector quintiles
        sector_quintiles = {}

        # Loop through each sector
        for sector in df['Sector'].unique():
            sector_df = df[df['Sector'] == sector]
            sector_df.sort_values(by=['Doc Sentiment Score'], ascending=[False], inplace=True)

            # Assign quintiles based on sentiment score
            sector_df['Quintile'] = pd.qcut(
                sector_df['Doc Sentiment Score'],
                q=5,
                labels=False,
                duplicates='drop'
            )
            sector_quintiles[sector] = sector_df

        # Initialize a list to store monthly sector returns
        monthly_sector_returns = []

        # Loop through each sector and calculate returns
        for sector, sector_df in sector_quintiles.items():
            top_df = sector_df[sector_df['Quintile'] == 4].reset_index(drop=True)
            bottom_df = sector_df[sector_df['Quintile'] == 0].reset_index(drop=True)

            top_returns = 0.0
            bottom_returns = 0.0
            num_stocks_to_buy = min(len(top_df), len(bottom_df))

            # Loop through each stock
            for i in range(num_stocks_to_buy):
                top_ticker = top_df.iloc[i]['Symbol']
                bottom_ticker = bottom_df.iloc[i]['Symbol']

                # Download stock data from Yahoo Finance
                top_stock_data = yf.download(top_ticker, start=start_date, end=end_date)
                bottom_stock_data = yf.download(bottom_ticker, start=start_date, end=end_date)

                # Calculate buy and sell prices
                top_buy_price = top_stock_data.iloc[0]['Open']
                top_sell_price = top_stock_data.iloc[-1]['Close']

                bottom_sell_price = bottom_stock_data.iloc[0]['Open']
                bottom_buy_price = bottom_stock_data.iloc[-1]['Close']

                # Calculate number of shares to buy and sell
                num_shares_to_buy = investment_per_stock / top_buy_price
                num_shares_to_sell = investment_per_stock / bottom_sell_price

                # Calculate returns for each stock
                top_returns += (top_sell_price - top_buy_price) * num_shares_to_buy / investment_per_stock
                bottom_returns += (bottom_sell_price - bottom_buy_price) * num_shares_to_sell / investment_per_stock

            # Store monthly returns for each sector
            monthly_sector_returns.append({
                'Sector': sector,
                'Buy': num_stocks_to_buy,
                'Sell': num_stocks_to_buy,
                'Returns': (top_returns + bottom_returns)
            })

        # Store monthly returns for all sectors
        sector_returns.append({
            'Year-Month': f"{year}-{month:02d}",
            'Returns': monthly_sector_returns
        })

# Calculate average monthly returns for all sectors
monthly_average_returns = []

for month_results in sector_returns:
    month_year = month_results['Year-Month']
    monthly_returns = month_results['Returns']

    total_returns = 0.0
    num_sectors = 0

    for sector_data in monthly_returns:
        total_returns += sector_data['Returns']
        num_sectors += 1

    # Calculate average returns for each month
    if num_sectors > 0:
        average_returns = total_returns / num_sectors
    else:
        average_returns = 0.0

    # Store monthly average returns
    monthly_average_returns.append({
        'Year-Month': month_year,
        'Average Returns': average_returns
    })

# Visualize monthly average returns for all sectors
months = [result['Year-Month'] for result in monthly_average_returns]
average_returns = [result['Average Returns'] for result in monthly_average_returns]

plt.figure(figsize=(12, 6))
plt.bar(months, average_returns, color='b')
plt.title('Monthly Average Returns for All Sectors')
plt.xlabel('Year-Month')
plt.ylabel('Average Returns')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

# Calculate and print the overall average returns
overall_average_returns = sum(average_returns) / len(average_returns)
print("Overall Average Returns:", overall_average_returns)

# Count positive and negative monthly returns
pos_count = sum(1 for result in average_returns if result >= 0)
neg_count = sum(1 for result in average_returns if result < 0)
print('Positive Counts:', pos_count)
print('Negative Counts:', neg_count)

# Save average returns to a file using pickle
import pickle
with open("average_returns.pkl", "wb") as file:
    pickle.dump(average_returns, file)

# Calculate the Sharpe ratio of your portfolio
portfolio_returns = average_returns  
risk_free_rate = 0.02  #  current risk-free rate (e.g., 10-year government bond yield) 2%

# Calculate the excess return of the portfolio (portfolio return - risk-free rate)
excess_return = [r - risk_free_rate for r in portfolio_returns]

# Calculate the standard deviation of the excess returns
portfolio_std_dev = np.std(excess_return)

# Calculate the Sharpe ratio
sharpe_ratio = (np.mean(excess_return) / portfolio_std_dev)

print("Portfolio Sharpe Ratio:", sharpe_ratio)


# Calculate the average returns for each sector as previously shown
sector_avg_returns = {}

for entry in sector_returns:
    year_month = entry['Year-Month']
    monthly_returns = entry['Returns']

    for sector_data in monthly_returns:
        sector = sector_data['Sector']
        returns = sector_data['Returns']

        if sector not in sector_avg_returns:
            sector_avg_returns[sector] = {'TotalReturns': 0, 'Count': 0}

        sector_avg_returns[sector]['TotalReturns'] += returns
        sector_avg_returns[sector]['Count'] += 1

# Calculate the average returns for each sector
for sector, data in sector_avg_returns.items():
    total_returns = data['TotalReturns']
    count = data['Count']

    if count > 0:
        average_returns = total_returns / count
    else:
        average_returns = 0  # Set average to 0 if no data available for a sector
    sector_avg_returns[sector]['AverageReturns'] = average_returns

# Sort sectors by average returns in descending order
sorted_sectors = sorted(sector_avg_returns.items(), key=lambda x: x[1]['AverageReturns'], reverse=True)

# Select the top 5 sectors
top_5_sectors = sorted_sectors[:5]

# Calculate the overall average returns for the top 5 sectors
overall_total_returns = sum(data['TotalReturns'] for _, data in top_5_sectors)
overall_count = sum(data['Count'] for _, data in top_5_sectors)
overall_average_returns = overall_total_returns / overall_count if overall_count > 0 else 0

# Print the average returns for the top 5 sectors and the overall average
for sector, data in top_5_sectors:
    print(f"Sector: {sector}, Average Returns: {data['AverageReturns']:.2f}")

print(f"Overall Average Returns for Top 5 Sectors: {overall_average_returns:.2f}")

