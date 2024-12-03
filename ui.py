
import dearpygui
from data_fetcher import fetch_stock_data
from historical_and_predicted_data_plotter import plot_with_predictions

'''
def fetch_and_display_data(sender, data):
    """
    Fetch stock data and display it alongside dummy predictions.
    """
    ticker = get_value("ticker_input")
    try:
        stock_data = fetch_stock_data(ticker)
        if "Close" not in stock_data.columns:
            raise ValueError("Invalid ticker or missing 'Close' column.")
        dummy_predictions = [x + 1 for x in stock_data["Close"]]
        plot_with_predictions(stock_data, dummy_predictions, ticker)
    except Exception as e:
        log_error(f"Error fetching data: {e}")
        
'''

def create_ui():
    """
    Create a simple GUI for fetching and displaying stock data using dearpygui.
    """
    with window("Stock Data Viewer"):
        add_text("Enter Stock Ticker:")
        add_input_text("ticker_input", label="")
        add_button("Fetch Data", callback=fetch_and_display_data)
        add_logger()
        
        start_dearpygui()
        
        
        
    
    
    
    

    
    

    


