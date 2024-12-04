import yfinance as yf

def fetch_stock_data(ticker):
    
    print(f"inside fetch function, ticker is {ticker}")
    """
    Fetch historical stock data for a given ticker using yfinance.

    Args:
        ticker (str): Stock ticker symbol.

    Returns:
        pandas.DataFrame: Historical stock data for the past 5 years.
    """
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period="5y")
    except Exception as e:
        print(f"error was {e}")
    return data

