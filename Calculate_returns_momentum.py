# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 10:11:53 2023

@author: Archana
"""



import pandas as pd
import os
from tqdm import tqdm



df= pd.read_csv(r'D:\Users\Archana\Downloads\FinBert_Sentiments\combined_data.csv').reset_index(drop = True)


df = df.drop(columns=['Unnamed: 0.1', 'Unnamed: 0', 'Date'])


df = df.rename(columns={'Accepted Date': 'Date'})

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

market_closing_time = datetime.strptime('16:00:00', '%H:%M:%S').time()

def calculate_returns(ticker_symbol, start_date, days_required):
    """
    Parameters
    ----------
    ticker_symbol : TYPE
        DESCRIPTION. Stock tickers
    date : TYPE
        DESCRIPTION. Date from which returns are calculated
    days_required : TYPE
        DESCRIPTION. 30 days, 90 days or 180 days for returns 
        
    Returns
    -------
    TYPE
        DESCRIPTION. Get the return from market close to market close
        start_date = date-1, if date.time is before market close on date

    """
    if start_date.time() < market_closing_time:
        reference_date = start_date.date() - timedelta(days=1)
    else:
        reference_date = start_date.date()

    end_date = reference_date + timedelta(days=days_required)
    try:
      stock_data  = yf.download(ticker_symbol, start=reference_date, end=end_date, progress=False)
      # do not return data for less than 3 days. Account for weekends
      if stock_data.index[-1].date()+timedelta(days=3) < end_date:
        return None
      start_price = stock_data["Adj Close"].iloc[0]
      end_price = stock_data["Adj Close"].iloc[-1]
      return ((end_price - start_price) / start_price)

    except Exception as e:

      print(f"Error calculating returns for {ticker_symbol}: {str(e)}")
      return None

def calculate_momentum(ticker_symbol, end_date, days_required):
    """

    Parameters
    ----------
    ticker_symbol : TYPE
        DESCRIPTION. Srock tickers
    end_date : TYPE
        DESCRIPTION. Date till which momentum is calculated 
    days_required : TYPE
        DESCRIPTION. 30 days, 90 days, or 365 days for momentum calculations. 

    Returns
    -------
    TYPE
        DESCRIPTION. If end_date is 30th and earnings call happened in the morning at 7:00 AM, the calculation
        of momentum is till the market close of previous date. If the earnings call is after market close, the caluclation 
        is till the end of that same date.

    """
    if end_date.time() < market_closing_time:
        end_date = end_date.date() - timedelta(days=1)

    reference_date = end_date - timedelta(days=days_required)

    try:

      stock_data  = yf.download(ticker_symbol, start=reference_date, end=end_date, progress=False)
      start_price = stock_data["Adj Close"].iloc[0]
      end_price = stock_data["Adj Close"].iloc[-1]
      return ((end_price - start_price) / start_price)

    except Exception as e:
      print(f"Error calculating momentum for {ticker_symbol}: {str(e)}")
      return None
  

df = df.dropna(subset=['Date'])

df['Date'] = pd.to_datetime(df['Date'])


for index, row in tqdm(df.iterrows(), desc="Processing rows"):
    ticker_symbol = row['Symbol']
    date = row['Date']

    one_month_return = calculate_returns(ticker_symbol, date, 30)
    three_month_return = calculate_returns(ticker_symbol, date, 90)
    six_month_return = calculate_returns(ticker_symbol, date, 180)

    one_month_momentum = calculate_momentum(ticker_symbol, date, 30)
    three_month_momentum = calculate_momentum(ticker_symbol, date, 90)
    twelve_month_momentum = calculate_momentum(ticker_symbol, date, 365)

    
    df.at[index, '1_month_forward_return'] = one_month_return
    df.at[index, '3_month_forward_return'] = three_month_return
    df.at[index, '6_month_forward_return'] = six_month_return

    df.at[index, '1_month_momentum'] = one_month_momentum
    df.at[index, '3_month_momentum'] = three_month_momentum
    df.at[index, '12_month_momentum'] = twelve_month_momentum
    
df.to_csv(r'D:\Users\Archana\Downloads\FinBert_Sentiments\complete_returns_date_corrected.csv')
    

def industry_average_1_month(group):
    numerator = (group['1_month_forward_return']*group['Market Cap']).sum()
    denominator = group['Market Cap'].sum()
    return numerator/denominator

result = df.groupby('Industry').apply(industry_average_1_month)

df['Relative_Returns_Industry_Average_1_month'] = df['1_month_forward_return'] - df['Industry'].map(result)

def industry_average_3_month(group):
    numerator = (group['3_month_forward_return']*group['Market Cap']).sum()
    denominator = group['Market Cap'].sum()
    return numerator/denominator

result = df.groupby('Industry').apply(industry_average_3_month)

df['Relative_Returns_Industry_Average_3_month'] =df['3_month_forward_return'] - df['Industry'].map(result)

def industry_average_6_month(group):
    numerator = (group['6_month_forward_return']*group['Market Cap']).sum()
    denominator = group['Market Cap'].sum()
    return numerator/denominator

result = df.groupby('Industry').apply(industry_average_6_month)

df['Relative_Returns_Industry_Average_6_month'] = df['6_month_forward_return'] - df['Industry'].map(result)

df.to_csv(r'D:\Users\Archana\Downloads\FinBert_Sentiments\relative_returns_date_corrected.csv')
#df.to_csv(r'D:\Users\Archana\Downloads\Earnings_Transcript\complete_returns_momentum_embeddings_.csv')


df.head()
