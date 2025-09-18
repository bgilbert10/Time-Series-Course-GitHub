#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ======================================================================
# DEALING WITH MULTIINDEXED COLUMNS FROM YFINANCE
# ======================================================================

# Import libraries
import yfinance as yf 
import pandas as pd
import time

# Set global float format to two decimal places
pd.options.display.float_format = '{:.2f}'.format

# ======================================================================
# SINGLE TICKER CASE
# ======================================================================

# Fetch Tesla data 
start_date = pd.to_datetime("2020-01-01")
end_date = pd.to_datetime("today")
tesla_data = yf.download("TSLA", start=start_date, end=end_date)
tesla_data.index = tesla_data.index.strftime('%Y-%m-%d')  # Format index as YYYY-MM-DD
print(tesla_data.head())

# Method 1: Flatten multi-level columns
# Flatten the multi-level columns to a single level
tesla_data.columns = tesla_data.columns.to_flat_index()
tesla_data.columns = ['_'.join(col).strip() for col in tesla_data.columns.values]
print(tesla_data.head(3))

# Method 2: Disable Multi-indexing
# Bypass multi-level columns
tesla_data = yf.download("TSLA", start=start_date, end=end_date, multi_level_index=False)
tesla_data.index = tesla_data.index.strftime('%Y-%m-%d')
print(tesla_data.head(3))

# Method 3: Drop unwanted levels 
# Drop the first level (Ticker)
tesla_data = yf.download("TSLA", start=start_date, end=end_date)
tesla_data.columns = tesla_data.columns.droplevel(1)
print(tesla_data.head(3))

# ======================================================================
# MULTIPLE TICKER CASE
# ======================================================================

# Fetch data for multiple tickers
tickers = ["TSLA", "AAPL", "MSFT"]
start_date = pd.to_datetime("2020-01-01")
end_date = pd.to_datetime("today")
data = yf.download(tickers, start=start_date, end=end_date)

# Display the first few rows
data.index = data.index.strftime('%Y-%m-%d')  # Format index
display(data.head())

# Can customize with group_by
data = yf.download(tickers, start=start_date, end=end_date, group_by='column')
display(data.head())
# versus
data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker')
display(data.head())
display(data['TSLA'].head())  # Access data for Tesla

# ======================================================================
# MULTIPLE TICKER CASE WITH A SINGLE DATAFRAME
# ======================================================================

# Method 1: Split by ticker (one call to yfinance for all tickers)
# Fetch data for multiple tickers
tickers = ["TSLA", "AAPL", "MSFT"]
start_date = pd.to_datetime("2020-01-01")
end_date = pd.to_datetime("today")

# Start timing
start_time = time.time()

# Download all data
data = yf.download(tickers, start=start_date, end=end_date)
data.index = data.index.strftime('%Y-%m-%d')

# Split into individual DataFrames for each ticker by slicing the MultiIndex
df_list = {ticker: data.xs(ticker, level=1, axis=1) for ticker in tickers}

# Stop timing
end_time = time.time()
print(f"\nMethod 1 runtime: {end_time - start_time:.2f} seconds")

# Example: Display Tesla's DataFrame
display(df_list["TSLA"].head())

# Method 2: Download separately for each ticker (multiple calls to yfinance)
# Start timing
start_time = time.time()

# Initialize an empty dictionary to store individual DataFrames
df_list = {}

# Download data for each ticker separately
for ticker in tickers:
    df = yf.download(ticker, start=start_date, end=end_date)
    df.index = df.index.strftime('%Y-%m-%d')
    df_list[ticker] = df

# Stop timing
end_time = time.time()
print(f"\nMethod 2 runtime: {end_time - start_time:.2f} seconds")

# Example: Display Tesla's DataFrame
display(df_list["TSLA"].head())