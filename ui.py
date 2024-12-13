import streamlit as st
from data_preprocessing import fetched_data_to_sequenced_data_with_scaler
from data_fetcher import fetch_stock_data
from data_preprocessing import scaling_and_reshaping_raw_data
from historical_and_predicted_data_plotter import plot_with_predictions
import torch
import pandas as pd
import numpy


st.title("Stock Predictor UI")
#title of the window

user_ticker = st.text_input("input a ticker")
#input field for user to input a ticker

from LSTM import train_lstm

if st.button("make prediction"):
    #button to make prediction
    
    if len(user_ticker) > 0:
        
        #fetches data and splits it into sequences for training
        X_sequenced, y_sequenced, scaler = fetched_data_to_sequenced_data_with_scaler(user_ticker)
        
        print("X_sequenced is:", X_sequenced)
        print("X_sequenced type is:", type(X_sequenced))
        print("X_sequenced shape is:", X_sequenced.shape)
        
        print("y_sequenced is:", y_sequenced)
        print("y_sequenced type is:", type(y_sequenced))
        print("y_sequenced shape is:", y_sequenced.shape)
        
        #making a trained model
        trained_lstm = train_lstm(X_sequenced, y_sequenced)
        
        print("training lstm done")
        
        #for a new prediction ie we actually don't know the next value, we get the final sequence
        print("fetching stock data:")
        
        fetched_data = fetch_stock_data(user_ticker)
        
        print("fetched stock data is:", fetched_data)
        print("fetched stock data type is:", type(fetched_data))
        print("fetched stock data shape is:", fetched_data.shape)
        
        fetched_data, _ = scaling_and_reshaping_raw_data(fetched_data)
        
        print("scaled fetched stock data is:", fetched_data)
        print("scaled fetched stock data type is:", type(fetched_data))
        print("scaled fetched stock data shape is:", fetched_data.shape)
        
        fetched_data = fetched_data[-50:]
        
        print("fin seq scaled fetched stock data is:", fetched_data)
        print("fin seq scaled fetched stock data type is:", type(fetched_data))
        print("fin seq scaled fetched stock data shape is:", fetched_data.shape)
        
        #this is where it is not reshaping
        fetched_data = fetched_data.reshape(1, -1, 1)
        
        print("reshaped numpy fin seq scaled fetched stock data is:", fetched_data)
        print("reshaped numpy fin seq scaled fetched stock data type is:", type(fetched_data))
        print("reshaped numpy fin seq scaled fetched stock data shape is:", fetched_data.shape)
        
        data_to_be_predicted_on = torch.tensor(fetched_data, dtype=torch.float32)
        
        print("tensored reshaped data is:", data_to_be_predicted_on)
        print("tensored reshaped data type is:", type(data_to_be_predicted_on))
        print("tensored reshaped data shape is:", data_to_be_predicted_on.shape)
        
        
        
        prediction = trained_lstm.forward(data_to_be_predicted_on)
        
        print("prediction is:", prediction)
        print("prediction type is:", type(prediction))
        print("prediction shape is:", prediction.shape)
        
        #going from torch tensor to numpy array as scaler only works w numpy
        prediction = prediction.detach().numpy()
        
        print("prediction is:", prediction)
        print("prediction type now is:", type(prediction))
        print("prediction shape now is:", prediction.shape)
        
        prediction = prediction.reshape(-1, 1)
        
        print("prediction shape now is:", prediction.shape)
        
        #data will be around values of 0 and 1 so need to scale back up to expected range
        prediction = scaler.inverse_transform(prediction)
        
        print("prediction is:", prediction)
        print("prediction type is:", type(prediction))
        print("prediction shape is:", prediction.shape)
        
        prediction = prediction.reshape(-1)
        
        print("prediction is:", prediction)
        print("prediction type is:", type(prediction))
        print("prediction shape is:", prediction.shape)
        
        #to be inserted into the graph as the historical data
        fetched_data = fetch_stock_data(user_ticker)
        
        print("fetched stock data is:", fetched_data)
        print("fetched stock data type is:", type(fetched_data))
        print("fetched stock data shape is:", fetched_data.shape)
        
        #plotting the data
        fig = plot_with_predictions(fetched_data[-5:], prediction, user_ticker)
        
        st.pyplot(fig)
        
    else:
        st.write("please enter ticker first")








