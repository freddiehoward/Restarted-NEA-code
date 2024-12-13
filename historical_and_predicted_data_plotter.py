import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_with_predictions(data, predictions, ticker):
    """
    Plot historical stock data for a given ticker alongside predicted data

    Inputs:
        data (pandas.DataFrame): DataFrame containing historical stock data in the form: index (date), data 1D.
        ticker (str): Stock ticker symbol.
        predictions(list/array): a list that contains dummy prediction values.

    Returns:
        None
    """

    
    plt.figure(figsize=(10, 5))
    
    print("predictions are:", predictions)
    print("len predictions is:", len(predictions))
    print("type predictions is:", type(predictions))
    
    plt.plot([1], predictions, marker='o', label=f"{ticker} Predicted Prices")
    
    print("data are:", data)
    print("len data is:", len(data))
    print("type data is:", type(data))
    
    data = np.array(data)
    
    print("data are:", data)
    print("len data is:", len(data))
    print("type data is:", type(data))
    
    plt.plot([-4,-3,-2,-1,0], data, label=f"{ticker} Historical Closing Price")
    
    plt.title(f"Historical and Predicted Stock Data for {ticker}")
    plt.xlabel("Time after current day (days)")
    plt.ylabel("Price")
    plt.legend()
    plt.grid()
    fig = plt.savefig("plot.png")
    return fig
    
    
