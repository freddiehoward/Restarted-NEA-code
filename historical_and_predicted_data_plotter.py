import matplotlib.pyplot as plt
from data_fetcher import fetch_stock_data

def plot_historical_data_with_predictions(data, predictions, ticker):
    """
    Plot historical stock data for a given ticker alongside predicted data

    Args:
        data (pandas.DataFrame): DataFrame containing historical stock data.
        ticker (str): Stock ticker symbol.
        predictions(list/array): a list that contains dummy prediction values.

    Returns:
        None
    """
    
    plt.figure(figsize=(10, 5))
    plt.plot(data.index[:len(predictions)], data['Close'][:len(predictions)], label=f"{ticker} Historical Closing Price")
    plt.plot(data.index[:len(predictions)], predictions, label=f"{ticker} Predicted Prices")
    plt.title(f"Historical and Predicted Stock Data for {ticker}")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid()
    plt.show()
    
print(plot_historical_data_with_predictions(fetch_stock_data("AAPL"), [240,250,255,275,289,286,277], "AAPL"))