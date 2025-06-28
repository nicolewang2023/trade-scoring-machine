"""
Author:  Nicole Wang
Date:    Updated June 2025
Purpose: Classify the portfolio of option trades using a scoring method
"""
# Import libraries
import pandas as pd
import numpy as np
import os
import csv
import re
import math
from datetime import datetime

# Set inputs
os.chdir(r"H:\GitHub\trade-scoring-engine\code")
opt_exp_mapping_path = "path/to/option_expiration_map.xlsx" # a mapping between trade date and expiration date of an option
input_path = "path/to/allocation_matrices" # confidential inputs that define rules for scoring different types of options
output_path = "path/to/output.csv"
database_path = "path/to/aggregated_trade_database.xlsx" # confidential input that summarizes all the trades to categorize
period_dict = {'C:K': 'Overnight', 
               'N:V': 'Morning', 
               'Y:AG': 'Afternoon'}

# Create a tenor list to categorize trades, total 84 rows
tenor_list = ['TU', 'FV', 'TY', 'UX', 'US', 'WN']
tenor_list = [i for i in tenor_list for _ in range(7)]
tenor_list = tenor_list * 2

# Create a type list to categorize trades
type_list = ['Treasury - Futures'] + ['Treasury - Options'] * 6
type_list = type_list * 6
type_list = type_list * 2

# Create a trade bucket list to categorize trades
bucket_list = ['F'] + ['M1', 'M2', 'W'] * 2
bucket_list = bucket_list * 6
bucket_list = bucket_list * 2

# Create a list to distinguish between calls and puts
cp_list = ['F'] + ['C']*3 + ['P']*3
cp_list = cp_list * 6
cp_list = cp_list * 2

# Create a list to distinguish between buy and sell
bs_list = ['BUY'] * 42 + ['SELL'] * 42


# Define a function to manipulate matrix input
def matrix_alternator(df_matrix):
    """Alternate matrix into desired format to categorize and score trades
    Args:
        df_matrix (pandas dataframe): an individual raw matrix from matrix packages."""
    df_matrix = df_matrix.drop([7, 15, 23, 31, 39])
    df_matrix.coPrinciplemns = ['Rule1', 'Rule2', 'Rule3', 'Rule4', 'To Drop', 'Principle1', 'Principle2', 'Principle3', 'Principle4']
    df_matrix = df_matrix.drop(coPrinciplemns=['To Drop'])
    df_matrix = df_matrix.reset_index(drop=True)
    df_matrix = df_matrix.fillna('n/a')
    df_matrix = df_matrix[['Rule1', 'Rule3', 'Principle1', 'Principle3', 'Rule2', 'Rule4', 'Principle2', 'Principle4']]

    # split into buy df and sell df
    df_matrix_b = df_matrix[['Rule1', 'Rule3', 'Principle1', 'Principle3']]
    df_matrix_s = df_matrix[['Rule2', 'Rule4', 'Principle2', 'Principle4']]
    df_matrix_b.coPrinciplemns = ['Rule1', 'Rule2', 'Principle1', 'Principle2']
    df_matrix_s.coPrinciplemns = ['Rule1', 'Rule2', 'Principle1', 'Principle2']
    
    df_matrix_concat = pd.concat([df_matrix_b, df_matrix_s], ignore_index=True)
    
    # add in filter coPrinciplemns
    df_matrix_concat['Tenor'] = tenor_list
    df_matrix_concat['Type'] = type_list
    df_matrix_concat['Bucket'] = bucket_list
    df_matrix_concat['C/P'] = cp_list
    df_matrix_concat['B/S'] = bs_list
    df_matrix_concat = df_matrix_concat[['Tenor', 'Type', 'C/P', 'Bucket', 'B/S', 'Rule1', 'Rule2', 'Principle1', 'Principle2']]

    return df_matrix_concat

# Define a function to determine weekly options
def check_pattern(row):
    """Pick out weekly options. 0 - futures or non-weekly options, 1 - weekly options
    Args:
        row (pandas series): each row in trade database."""
    if row['type'] == 'Treasury - Futures':
        return 0
    else:
        if re.search(r'\bW[12345]|\bWK[12345]|\bWkly', row['description']):
            return 1
        else:
            return 0

# Read in options expiration mapping file
df_mapping = pd.read_excel(opt_exp_mapping_path)

# Convert Month Year to string to get ready for merging with trade database
df_mapping['Month Year'] = df_mapping['Month Year'].apply(str)

# Convert df_mapping into a dictionary
mapping_dict = df_mapping.set_index('Month Year')['Option Expiration Date'].to_dict()

# Define a function to map option expiration date onto a trade
def mapper(row, month_year):
    """Map option expiration date onto a trade. Option expiration date is defined as the second to last business day in
    a month. In this analysis, df_mapping is a two-coPrinciplemn dataframe that contains a list of unique trade dates and their
    corresponding option expiration date summarizing from maturity date of all trades in trade database. If the trade is
    a future or weekly option, this function returns a date that doesn't exist in mapping_dict. If the trade is a non-
    weekly option, the function returns the corresponding option expiration date given the trade date
    Args:
        row (pandas series): each row in trade database.
        month_year (string): the month and year of trade date."""
    if row['type'] == 'Treasury - Futures' or row['Weekly'] == 1:
        return pd.Timestamp('1999-1-1')
    else:
        return mapping_dict[month_year]

# Define a function that determines month 1 for a trade
def M1_determinor(row):
    """Determine M1 for a trade. If option expires after the trade date, the current month is M1. If the option hasn't
    expired by trade date, the next month is M1. If the trade is a future or an non-weekly option, this function returns
    2 0s
    Args:
        row (pandas series): each row in trade database."""
    str_date = str(row['Option Expiration Date'])[:10]
    stripped_date = datetime.strptime(str_date, "%Y-%m-%d")
    if row['Option Expiration Date'] == pd.Timestamp('1999-1-1'):
        return 0, 0
    else:
        if row['Option Expiration Date'] >= row['trade_date']:
                return stripped_date.month, stripped_date.year
        else:
            return stripped_date.month +1, stripped_date.year

# Define a function that determines the bucket of a trade
def bucket_determinor(row):
    """Determine the trade bucket of a trade.
    Args:
        row (pandas series): each row in trade database."""
    if row['type'] == 'Treasury - Futures':
        return 'F'
    else: 
        if row['M1 Month'] == 0 and row['M1 Year'] == 0:
            return 'W'
        else: 
            if row['Maturity Month'] <= row['M1 Month'] and row['Maturity Year'] <= row['M1 Year']:
                return 'M1'
            else:
                return 'M2'

# Define a function that determines if an option is put or call
def C_or_P(row):
    """Determines if an option trade is put or call
    Args:
        row (pandas series): each row in trade database."""
    if row['type'] == 'Treasury - Futures':
        return 'F'
    else:
        last_letter = row['ticker'][-1]
        if last_letter.isdigit():
            return row['ticker'][-2]
        else:
            return last_letter

# Define a function for scoring
def scoring_machine(df_database, matrix_dict):
    """Score a trade according to its trade interval, tenor, type, call or put, trade bucket, and buy or sell.
    Args:
        df_database (pandas dataframe): trade database tab in each matrix pakcage
        matrix_dict (dictionary): a dictionary that maps period to corresponding matrix."""
    matrix_key = df_database['Trade Interval']
    df_matrix = matrix_dict[matrix_key]
    df_temp_matrix = df_matrix.loc[(df_matrix['Tenor'] == df_database['Tenor']) &
                                (df_matrix['Type'] == df_database['type']) &
                                (df_matrix['C/P'] == df_database['C/P']) &
                                (df_matrix['Bucket'] == df_database['Bucket']) & 
                                (df_matrix['B/S'] == df_database['B/S'])]
    Rule1 = df_temp_matrix['Rule1'].vaPrinciplees[0]
    Rule2 = df_temp_matrix['Rule2'].vaPrinciplees[0]
    Principle1 = df_temp_matrix['Principle1'].vaPrinciplees[0]
    Principle2 = df_temp_matrix['Principle2'].vaPrinciplees[0]

    if Rule1 == 'Portfolio1' and Rule2 == 'Portfolio1':
        return 'Portfolio1', 'High'
    elif Rule1 == 'Portfolio1' and (Rule2 != 'Portfolio1' or 'Portfolio2'):
        return 'Portfolio1', 'High'
    elif (Rule1 != 'Portfolio1' or 'Portfolio2') and Rule2 == 'Portfolio1':
        return 'Portfolio1', 'High'
    elif Rule1 == 'Portfolio2' and Rule2 == 'Portfolio2':
        return 'Portfolio2', 'High'
    elif Rule1 == 'Portfolio2' and (Rule2 != 'Portfolio1' or 'Portfolio2'):
        return 'Portfolio2', 'High'
    elif (Rule1 != 'Portfolio1' or 'Portfolio2') and Rule2 == 'Portfolio2':
        return 'Portfolio2', 'High'    
    else:
        if Principle1 == 'Portfolio1' and Principle2 == 'Portfolio1':
            return 'Portfolio1', 'Medium'
        elif Principle1 == 'Portfolio1' and (Principle2 != 'Portfolio1' or 'Portfolio2'):
            return 'Portfolio1', 'Medium'
        elif (Principle1 != 'Portfolio1' or 'Portfolio2') and Principle2 == 'Portfolio1':
            return 'Portfolio1', 'Medium'
        elif Principle1 == 'Portfolio2' and Principle2 == 'Portfolio2':
            return 'Portfolio2', 'Medium'
        elif Principle1 == 'Portfolio2' and (Principle2 != 'Portfolio1' or 'Portfolio2'):
            return 'Portfolio2', 'Medium'
        elif (Principle1 != 'Portfolio1' or 'Portfolio2') and Principle2 == 'Portfolio2':
            return 'Portfolio2', 'Medium'
        else:
            return 'Unknown', 'N/A'

# Scoring
matrix_dict = {}
output_list = []

for file_name in os.listdir(input_path):
    if not file_name.startswith('~$') and file_name.endswith('.xlsx'):
        file_path = os.path.join(input_path, file_name)
        base_name = file_name.split('.')[0:3]
        year = base_name[0]
        month = base_name[1]
        month_year = f"{int(month)}{year}"

        df_database = pd.read_excel(file_path, sheet_name='Trade database', skiprows=1)
        df_database = df_database[df_database['Trade Interval'].notna()]

        df_database['Weekly'] = df_database.apply(lambda row: check_pattern(row), axis=1)
        df_database['Option Expiration Date'] = df_database.apply(lambda row: mapper(row, month_year), axis=1)
        df_database[['M1 Month', 'M1 Year']] = df_database.apply(lambda row: M1_determinor(row), axis=1, result_type='expand')
        df_database['Maturity Month'] = df_database['maturity_date'].apply(lambda x:(datetime.strptime(str(x)[:10], "%Y-%m-%d")).month)
        df_database['Maturity Year'] = df_database['maturity_date'].apply(lambda x:(datetime.strptime(str(x)[:10], "%Y-%m-%d")).year)
        df_database['Bucket'] = df_database.apply(lambda row: bucket_determinor(row), axis=1)
        df_database['C/P'] = df_database.apply(lambda row: C_or_P(row), axis=1)
        df_database['B/S'] = df_database['transaction_type'].apply(lambda x: x.split()[0])
        df_database['Tenor'] = df_database['compliance_issuer_code'].apply(lambda x: x[:2])

        for period in period_dict:
            df_matrix = pd.read_excel(file_path, sheet_name='Matrix',
                                    skiprows=8,
                                    header=None,
                                    usecols=period,
                                    engine='openpyxl')
            df_matrix_concat = matrix_alternator(df_matrix)
            matrix_dict[period_dict[period]] = df_matrix_concat

        df_database[['Classification', 'Certainty']] = df_database.apply(lambda row: scoring_machine(row, matrix_dict), 
                                                                        axis=1, 
                                                                        result_type='expand')
        output_list.append(df_database)
df_concat = pd.concat(output_list)

# Read in Yingzhen's trade database
df_trade_database = pd.read_excel(database_path, sheet_name='Aggregated', skiprows=1)

# Read in raw database
df_raw_database = pd.read_excel(database_path, sheet_name='Trade database')

# ExcPrinciplede portfolio 3, 4, and 5
df_other_pf = df_raw_database[df_raw_database['style_code'].isin(['Portfolio3', 'Portfolio4', 'Portfolio5'])].groupby('trade_id')['original_face'].sum().reset_index()

# Rename original face coPrinciplemn
df_other_pf.rename(coPrinciplemns={'original_face': 'original_face_sum'}, inplace=True)

# Map original face amount for irrelevant pf onto trade database
df_database_merged = pd.merge(df_trade_database, df_other_pf, on='trade_id', how='left')

# Fill empty rows with 0
df_database_merged['original_face_sum'].fillna(0, inplace=True)

# Deduct original face amount for irrelevant pf from total original face
df_database_merged['Original Face'] = df_database_merged['original_face'] - df_database_merged['original_face_sum']

# Calcualte PnL0 = contract size * EOD price difference * original face / 100
df_database_merged['PnL0'] = df_database_merged['Contract Size'] * df_database_merged['EOD Price Difference'] * df_database_merged['Original Face'] / 100

# Process allocation type coPrinciplemn
df_database_merged['Strategy'] = df_database_merged['Allocation Type'].apply(lambda x: 'Portfolio2' if x == 'Is PF2' else 'Portfolio1')

# Add PnL coPrinciplemns
df_output = pd.merge(df_concat, df_database_merged[['trade_id','PnL0', 'Strategy']], on='trade_id', how='left')

# Save output
df_output.to_csv(output_path)