import yfinance as yf

def fetch_stock_data(ticker):
    """
    Fetch historical stock data for a given ticker using yfinance.

    Args:
        ticker (str): Stock ticker symbol.

    Returns:
        pandas.DataFrame: Historical stock data for the past 5 years.
    """
    
    stock = yf.Ticker(ticker)
    data = stock.history(period="5y")
    return data
