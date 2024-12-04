import dearpygui.dearpygui as dpg
from data_fetcher import fetch_stock_data
from historical_and_predicted_data_plotter import plot_with_predictions


dpg.create_context()
dpg.create_viewport(title='Custom Title', width=600, height=300)



with dpg.window(tag="Primary Window"):
    ticker_input = dpg.add_input_text(label="input a ticker")
    make_prediction_button = dpg.add_button(label="make prediction", callback=lambda: get_ticker_from_input_field())
    #lambda so function in callback doesn't just immediately run but waits until button pressed
    
def get_ticker_from_input_field():
    
    input_value = dpg.get_value(ticker_input)
    
    print(input_value)
    
    
def make_prediction_ui_function():
    
    ticker = get_ticker_from_input_field()
    
    stock_data = fetch_stock_data(ticker)
    
    #train_LSTM(stock_data[:-50])
    
    #predictions = run_LSTM(stock_data[-50:])
    
    predictions = [5,6,7]
    
    plot_with_predictions(stock_data, predictions)

'''
input stock ticker
get input
fetch historical data on stock ticker
train lstm
run lstm
run plotter, passing in lstm predictions, historical data 
'''

dpg.setup_dearpygui()
dpg.show_viewport()
dpg.set_primary_window("Primary Window", True)
dpg.start_dearpygui()
dpg.destroy_context()

#create context
#create viewport
#setup dearpygui
#show viewport
#start dearpygui
#destroy context


