import streamlit as st
from data_fetcher import fetch_stock_data
from historical_and_predicted_data_plotter import plot_with_predictions


st.title("Stock Predictor UI")
#title of the window

st.write("hello, world")
#display a message

user_ticker = st.text_input("input a ticker")
#input field for user to input a ticker

if st.button("make prediction"):
    #button to make prediction
    
    if len(user_ticker) > 0:
        data = fetch_stock_data(user_ticker)
        #train_LSTM(data[:-50])
        #predictions = run_LSTM(data[-50:])
        predictions = [x+50 for x in data['Close']]
        fig = plot_with_predictions(data, predictions, user_ticker)
        st.pyplot(fig)
        
    else:
        st.write("please enter ticker first")








