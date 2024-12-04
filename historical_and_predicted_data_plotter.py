import matplotlib.pyplot as plt
from data_fetcher import fetch_stock_data
import pandas as pd

def plot_with_predictions(data, predictions, ticker):
    """
    Plot historical stock data for a given ticker alongside predicted data

    Args:
        data (pandas.DataFrame): DataFrame containing historical stock data.
        ticker (str): Stock ticker symbol.
        predictions(list/array): a list that contains dummy prediction values.

    Returns:
        None
    """
    try:
    
        predicted_index = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=len(predictions), freq='D')
        
        plt.figure(figsize=(10, 5))
        plt.plot(data.index, data['Close'], label=f"{ticker} Historical Closing Price")
        plt.plot(predicted_index, predictions, label=f"{ticker} Predicted Prices")
        plt.title(f"Historical and Predicted Stock Data for {ticker}")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.grid()
        fig = plt.savefig("plot.png")
        
    except:
        print("Error plotting data")
        
    return fig