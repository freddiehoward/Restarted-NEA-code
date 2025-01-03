import yfinance as yf

def fetch_stock_data(ticker):
    
    """
    Fetch historical stock data for a given ticker using yfinance.

    Args:
        ticker (str): Stock ticker symbol.

    Returns:
        pandas.DataFrame: Historical stock data for the past year.
    """
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period="1y")
        data = data['Close']
    except Exception as e:
        print(f"error was {e}")
        return None
        
    #data is a pandas dataframe    
    return data

