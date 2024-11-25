#######################################################################################################################################
#
# Title: Portfolio Builder and Updater with Python
#
# Authoer: Nick Hosler 
# Sources: 
# 1. randerson112358's Medium Post On using Python for Portfolio Builing: 
# https://randerson112358.medium.com/python-for-finance-portfolio-optimization-66882498847
# 2. tagoma's butiful soup based function to extract most up to date list of S&P500 constituents known to date from Wikipedia
#
# Purpose: Create a sharpe ratio maximizing portfolio based on a given amount of capital available from S&P500 Constituents or a list
# of the users choosing. 
# May upgrade in the future to leverage a report system that can read from a previous report to generate a list of actions that should
# be taken to modify the portfolios current holdings to reflect the Sharpe ratio maximizing portfolio. Save past holdings and return
# a list of actions to be taken to update the model portfolio. It is a rough script and could be cleaned further, but was built to 
# increase my own understanding, and as such hasn't been cleaned to a deliverable level of quality. Hope you find this interesting!
#
######################################################################################################################################

from urllib import request
import os
from pathlib import Path
from bs4 import BeautifulSoup
import datetime
from datetime import datetime
import time
import dateutil.relativedelta as dr
import pandas as pd
from pandas_datareader import data as web
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
import warnings


import yfinance as yfin
import pandas_datareader as pdr
yfin.pdr_override()

# Functions 

# function to get list of all S&P500 Constituents from Wikipedia list (most up to date available publicly)
def get_constituents():
    # URL request, URL opener, read content
    req = request.Request('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    opener = request.urlopen(req)
    content = opener.read().decode() # Convert bytes to UTF-8

    soup = BeautifulSoup(content)
    tables = soup.find_all('table') # HTML table we actually need is tables[0] 

    external_class = tables[0].findAll('a', {'class':'external text'})

    tickers = []

    for ext in external_class:
        if not 'reports' in ext:
            tickers.append(ext.string)
    
    # adding this loop to change "." to "-" for yahoo finance API  
    for i in range(len(tickers)):
        tickers[i] = tickers[i].replace(".","-")
       
    return tickers

# function to compute the arbitrary weights based on the number of constiutents in the asset list
def generate_arbitrary_weights(asset_list):
    # use the function to assign the available assets
    # assign arbitrary weights based on the number of constituents, this way it will update based on any additions to the SP500
    even_weights = 1/(len(asset_list))
    # generate array to populate
    weights = np.array(even_weights)
    
    for i in range((len(asset_list)-1)):
        weights = np.append(weights,even_weights)
        
    return weights
    
def read_asset_list_pricing_data(asset_list,df,stock_start,today):
    print("Procurring data for custom list or, S&P500 member firms, this may take several minutes...")
    for stock in asset_list:
        try:
            print("Getting data for: "+stock+"...")
            df[stock] = yfin.download(stock, start=stock_start,end=today)['Adj Close']
            
        except:
            print(stock + ' missing dates')
            pass
    return df

def plot_securities(securities):
        
    # Start creating views of the data (likely a mess)
    title = 'Trended Portfolio Adj. Close Price of New Portfolio Holdings'
    
    #Create and plot chart
    plt.figure(figsize=(12.2,4.5)) #12.2IN W 4.5IN H
    
    for i in securities.columns.values:
        plt.plot(securities[i],label = i)  #plt.plot( X-Axis , Y-Axis, line_width, alpha_for_blending,  label)
        
    plt.title(title)
    plt.xlabel('Date',fontsize=18)
    plt.ylabel('Adj. Price USD ($)',fontsize=18)
    plt.legend(securities.columns.values,loc='upper left')
    plt.draw()
    plt.show()
    return
    
def discrete_allocations(df,weights,new_portfolio_value):
    latest_prices = get_latest_prices(df)
    da = DiscreteAllocation(weights, latest_prices, total_portfolio_value=new_portfolio_value)
    allocation, leftover = da.lp_portfolio()
    # some re-formatting for the latest_prices series so that I can use it how I intend
    latest_prices = pd.Series.to_frame(latest_prices)
    latest_prices.reset_index(inplace=True)
    as_of_date = str(latest_prices.columns[1]).split(" ")[0]
    latest_prices.columns = ["Ticker","Price as of close "+as_of_date]

    return allocation, leftover, latest_prices, as_of_date

def task_list(do_df):
    print("\n")
    print("Steps to update portfolio:")
    if len(do_df) != 0:
        for i in range(len(do_df)):
            time.sleep(.1)
            if do_df.iloc[i][4] > 0:
                print("Purchase "+str(int(do_df.iloc[i][4]))+" more unit(s) of " + str(do_df.iloc[i][0]) + " stock." )
            elif do_df.iloc[i][4] < 0:
                    print("Sell "+str(int(abs(do_df.iloc[i][4])))+"  unit(s) of " + str(do_df.iloc[i][0]) + " stock.")       
    else:
        print("No changes need to be made to the portfolio at this time")
        
def get_portfolio_value():
     # Need new total value of total portfolio, any prior holdings plus additional contributions
    while True:
        try:
            new_portfolio_value = int(input("Enter the total value of your portfolio holdings rounded to the nearest whole dollar (i.e. 10000): "))
            break
        except:
            print("That's not a valid option! Please enter an ")
    return new_portfolio_value

def update_holdings(cwd,allocation,leftover,latest_prices,df):
    print("Discrete allocations:" + "\n")
    allocation_list = list()
    for k,v in allocation.items():
        time.sleep(.25)
        print("You should hold "+str(v)+" units of "+ k)
        allocation_list.append(k)
        
    time.sleep(.25)
    print("Funds remaining after above holdings: ${:.2f}".format(leftover))
    securities = df.loc[:,allocation_list]
       
    # all for comparing
    try:
        # Time Stamp for portfolio records - epoch format (seconds elapsed since Jan 1 1970) 
        timestamp = str(round(time.time()))
        
        latest_prices_df = latest_prices.set_index('Ticker')
        allocations_df = pd.DataFrame(list(allocation.items()),columns=['Ticker','Allocation'])
        allocations_df = allocations_df.set_index('Ticker') 
        
        combined = latest_prices_df.join(allocations_df)
        combined = combined.fillna(0)
        combined = combined.reset_index()
        
        allocations_df.columns = ['New Allocations']
        # Pull in the most recent historical allocations file if it exists, wrap in try catch
        paths = sorted(Path(cwd+"\\02 Prior Holdings").iterdir(), key=os.path.getmtime)[-1]
        prior_portfolio_holdings = pd.read_csv(paths.__str__())
        #write out the combined data with the timestamp
        combined.to_csv(cwd+"\\02 Prior Holdings\\Portfolio_Allocations_"+timestamp+".csv",index=False)
        
        prior_portfolio_holdings = prior_portfolio_holdings.set_index('Ticker')
        # join the new allocations onto the old allocations
        changes_df = prior_portfolio_holdings.join(allocations_df,how='outer').fillna(0)
        changes_df = changes_df.reset_index()
        changes_df['Changes'] =  changes_df['New Allocations'] - changes_df['Allocation']
        
        do_df = changes_df[changes_df['Changes'] != 0]
        #execute task list
        task_list(do_df)
    except: 
            print("No Portfolio History to compare against for update, no update required. \n"+
                  "Current generated Holdings Have been saved to \\02 Prior Holdings.")
            #write out the combined data with the timestamp
            combined.to_csv(cwd+"\\02 Prior Holdings\\Portfolio_Allocations_"+timestamp+".csv",index=False)
            
    plot_securities(securities) 
            
def get_asset_list():
        ask = True
        while ask ==True: 
            try:
                 asset_response = str(input("\nWould you like to like build a portfolio from the S&P500 or a custom list?\n"+
                                          "Type SP500 or CUSTOM to proceed.\n"))
                 if asset_response == "SP500":
                    # Use the get_constituents() function to get the active list of S&P500 member firms maintained by Wikipedia
                    asset_list = get_constituents()
                    ask = False
                    
                 elif asset_response == "CUSTOM":
                        asset_list = input("Please enter a list of desired tickers seperated by a comma.\n"+
                                           "For example: TSLA,AMZN,JPM,GOOG.\n")
                        asset_list = asset_list.split(',')
                        ask = False
                 else:
                     print("That's not a valid option! Please enter SP500 or CUSTOM.")
            except:
                print("That's not a valid option! Please enter SP500 or CUSTOM.")
        return asset_list
    
    
# Begin Main Script  

def main():
    print("Welcome to the portfolio optimizer 1.0!")
    print("The purpose of this program is to show you the difference between investing in an arbitrarily \n"+
          "weighted portfolio consisting of S&P500 member firms, and a Sharpe Ratio optimized risk-adjusted \n"+
          "return maximizing portfolio. It will store your last holdings locally and give you a task list to adjust \n"+
          "your new portfolio when you re-run the script. I hope you find this code and its purpose as insightful as I did!")
    
    # Ask user for portfolio value
    new_portfolio_value = get_portfolio_value()
    
    asset_list = get_asset_list()
    
    # Generate the arbitrary weights based on the number of member firms
    arbitrary_weights = generate_arbitrary_weights(asset_list)
    
    # Get the date we want to start considering from 
    stock_start = '2015-01-01'
    
    # Get the date we want to stop considerations, current day being most practical
    today = datetime.today().strftime('%Y-%m-%d')
    
    #Create a df to hold the adjusted close price of the stocks in the SP500
    df = pd.DataFrame()
    
    # get the close price into the dataframe using the read_asset_list_pricing_data
    #asset_list = ['AAPL','AMZN']
    df = read_asset_list_pricing_data(asset_list,df,stock_start,today)
    #print(df)
    
    # assign the securities to the dataframe for use in visualization
    #securities = df # currently unused
    # not the most efficient use, better for only securities selected by optimizer
    #plot_securities(securities)
    
    # calculate simple daily returns, new_price/old_price - 1
    returns = df.pct_change()
    returns
    
    # create the annual covariance matrix
    cov_matrix_annual = returns.cov() * 252
    cov_matrix_annual
    
    # Calculate the annual variance of the portfolio
    # Expected portfolio variance= WT * (Covariance Matrix) * W
    variance = np.dot(arbitrary_weights.T,np.dot(cov_matrix_annual,arbitrary_weights))
    variance
    
    # Calculate the expected portfolio volatility
    volatility = np.sqrt(variance)
    volatility
    
    # Caclulate simple annual returns
    simple_annual_returns = np.sum(returns.mean()*arbitrary_weights)*252
    simple_annual_returns
    
    # putting the numbers into a more human interpretable format
    perc_var = str(round(variance*100,3)) + '%'
    perc_vol = str(round(volatility*100,3)) + '%'
    perc_ret = str(round(simple_annual_returns*100,3)) + '%'
    time.sleep(.1)
    
    print("\nExpected Annual Returns with arbitrary weights: " + perc_ret)
    time.sleep(.1)
    print("Annual Volatility/Std. Deviation & Risk with arbitrary weights: " + perc_vol)
    time.sleep(.1)
    print("Annual Variance with arbitrary weights: " + perc_var)
    time.sleep(.1)
    
    # optimizing the portfolio
    mu = expected_returns.mean_historical_return(df) #returns.mean()*252
    S = risk_models.sample_cov(df) # sample cov for matrix
    
    # Buyild the Efficient Frontier
    ef = EfficientFrontier(mu,S)
    # Get the raw weights that would maximize the sharpe ratio
    weights = ef.max_sharpe() 
    
    cleaned_weights = ef.clean_weights()
    
    #print(cleaned_weights)
    print("\nSharpe Optimized Portfolio Performance and Allocations: ")
    time.sleep(.1)
    ef.portfolio_performance(verbose=True)

    # getting discrete allocations and historical holdings
    # Get the current working directory
    cwd = os.getcwd()
    
    # functionalize with discrete_allocations(df,weights) 0-allocations, 1-remainder, 2-latest prices, 3-as of date
    data = discrete_allocations(df,cleaned_weights,new_portfolio_value)

    
    
    
    
    ################### UNPACKING UPDATE HOLDINGS TO REVISE ###############################

    # update holdings - save a copy of holdings and provides instructions to update
    #update_holdings(cwd,data[0],data[1],data[2],df)
    
  
    
    ####def update_holdings(cwd,allocation,leftover,latest_prices,df):
    
    #adding manual declarations for testing
    allocation = data[0]
    leftover = data[1]
    latest_prices = data[2]
    
    print("Discrete allocations:" + "\n")
    allocation_list = list()
    for k,v in allocation.items():
        time.sleep(.25)
        print("You should hold "+str(v)+" units of "+ k)
        allocation_list.append(k)
        
    time.sleep(.25)
    print("Funds remaining after above holdings: ${:.2f}".format(leftover))
    securities = df.loc[:,allocation_list]
       
    # all for comparing
    try:
        # Time Stamp for portfolio records - epoch format (seconds elapsed since Jan 1 1970) 
        timestamp = str(round(time.time()))
        
        latest_prices_df = latest_prices.set_index('Ticker')
        allocations_df = pd.DataFrame(list(allocation.items()),columns=['Ticker','Allocation'])
        allocations_df = allocations_df.set_index('Ticker') 
        
        combined = latest_prices_df.join(allocations_df)
        combined = combined.fillna(0)
        combined = combined.reset_index()
        
        allocations_df.columns = ['New Allocations']
        # Pull in the most recent historical allocations file if it exists, wrap in try catch
        paths = sorted(Path(cwd+"\\02 Prior Holdings").iterdir(), key=os.path.getmtime)[-1]
        prior_portfolio_holdings = pd.read_csv(paths.__str__())
        #write out the combined data with the timestamp
        combined.to_csv(cwd+"\\02 Prior Holdings\\Portfolio_Allocations_"+timestamp+".csv",index=False)
        
        prior_portfolio_holdings = prior_portfolio_holdings.set_index('Ticker')
        # join the new allocations onto the old allocations
        changes_df = prior_portfolio_holdings.join(allocations_df,how='outer').fillna(0)
        changes_df = changes_df.reset_index()
        changes_df['Changes'] =  changes_df['New Allocations'] - changes_df['Allocation']
        
        do_df = changes_df[changes_df['Changes'] != 0]
        #execute task list
        task_list(do_df)
    except: 
            print("No Portfolio History to compare against for update, no update required. \n"+
                  "Current generated Holdings Have been saved to \\02 Prior Holdings.")
            #write out the combined data with the timestamp
            combined.to_csv(cwd+"\\02 Prior Holdings\\Portfolio_Allocations_"+timestamp+".csv",index=False)
            
    #plot_securities(securities) 
     
    ########### def plot_securities(securities): ############
    """
    # Start creating views of the data (likely a mess)
    title = 'Trended Portfolio Adj. Close Price of New Portfolio Holdings'
    
    #Create and plot chart
    plt.figure(figsize=(12.2,4.5)) #12.2IN W 4.5IN H
    
    for i in securities.columns.values:
        plt.plot(securities[i],label = i)  #plt.plot( X-Axis , Y-Axis, line_width, alpha_for_blending,  label)
        
    plt.title(title)
    plt.xlabel('Date',fontsize=18)
    plt.ylabel('Adj. Price USD ($)',fontsize=18)
    plt.legend(securities.columns.values,loc='upper left')
    plt.draw()
    plt.show()
    """
    
    # New GPT version
    
    """
    Plots the adjusted close prices of securities from a DataFrame.
    
    Parameters:
    securities (pd.DataFrame): DataFrame where columns are securities, and rows are dates with price data.
    """
    # Ensure proper formatting of the title
    title = 'Trended Portfolio Adj. Close Price of New Portfolio Holdings'

    # Validate input to avoid runtime errors
    if securities.empty:
        raise ValueError("The 'securities' DataFrame is empty. Ensure it contains data.")

    if not hasattr(securities.index, 'name') or securities.index.name != 'Date':
        raise ValueError("Ensure the DataFrame index is properly set as 'Date'.")

    # Create the figure and set the size
    plt.figure(figsize=(12.2, 4.5))  # Ensure size provides good readability

    # Iterate through each column in the DataFrame and plot the data
    for column in securities.columns:
        if securities[column].isnull().all():
            print(f"Warning: Column '{column}' contains all NaN values and will be skipped.")
            continue  # Skip plotting columns with all missing values
        plt.plot(securities.index, securities[column], label=column)  # Correctly assign X (index) and Y (column)

    # Add title and axis labels with improved readability
    plt.title(title, fontsize=16)  # Increase font size for clarity
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Adj. Price USD ($)', fontsize=14)

    # Adjust the legend to handle potential overlapping issues
    plt.legend(loc='upper left', fontsize=10)  # Smaller font size for a clearer legend

    # Show the grid for better visual alignment
    plt.grid(True)

    # Ensure the plot is drawn properly before showing
    plt.tight_layout()  # Prevent overlapping of labels and legend
    plt.show()
    
    ########## END TESTING #################
    
    
    
    
    
    
    
    
    again = True
    while again == True: 
        try:
             response = str(input("Would you like to rebalance the portfolio for a different total portfolio value? (Y/N) \n"))
             if response == "Y":
                #Get new portfolio value
                new_portfolio_value = get_portfolio_value()
                # functionalize with discrete_allocations(df,weights) 0-allocations, 1-remainder, 2-latest prices
                data = discrete_allocations(df,cleaned_weights,new_portfolio_value)
                
                # update holdings - save a copy of holdings and provides instructions to update
                update_holdings(cwd,data[0],data[1],data[2],df)
             elif response == "N":
                 again = False
                 print("Thank you for using the portfolio optimizer 1.1")
             else:
                 print("That's not a valid option! Please enter Y or N.")
        except:
            print("That's not a valid option! Please enter Y or N.")
            
# main executable from command line    
if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
                
            
    
