import dearpygui.dearpygui as dpg
from data_fetcher import fetch_stock_data
from historical_and_predicted_data_plotter import plot_with_predictions


dpg.create_context()
dpg.create_viewport(title='Custom Title', width=600, height=300)


with dpg.window(label="Example Window"):
    dpg.add_text("Hello, world")
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()



        
    
    
    
    

    
    

    


