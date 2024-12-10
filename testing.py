from data_fetcher import fetch_stock_data
from data_preprocessing import scaling_and_reshaping_raw_data
from data_preprocessing import sequence_scaled_and_reshaped_data
from data_preprocessing import split_sequenced_data_into_test_and_train
from data_preprocessing import fetched_data_to_sequenced_data_without_scaler
from LSTM import LSTM


def test_321():
    test_dataframe = fetch_stock_data("AAPL")
    print(scaling_and_reshaping_raw_data(test_dataframe))
    return 0
    #should output data scaled between 0 and 1
    
def test_322():
    test_dataframe = fetch_stock_data("AAPL")
    test_dataframe, _ = scaling_and_reshaping_raw_data(test_dataframe)
    X_sequenced, y_sequenced = sequence_scaled_and_reshaped_data(test_dataframe)
    print("X_sequenced shape is", X_sequenced.shape)
    print("y_sequenced shape is", y_sequenced.shape)
    
def test_323():
    test_dataframe = fetch_stock_data("AAPL")
    test_dataframe, _ = scaling_and_reshaping_raw_data(test_dataframe)
    X_sequenced, y_sequenced = sequence_scaled_and_reshaped_data(test_dataframe)
    X_train, X_test, y_train, y_test = split_sequenced_data_into_test_and_train(X_sequenced, y_sequenced)
    print("shape X_train is:", X_train.shape)
    print("shape y_train is:", y_train.shape)
    print("shape X_test is:", X_test.shape)
    print("shape y_test is:", y_test.shape)
    
'''
def test_324():
    test_dataframe = fetch_stock_data("AAPL")
    test_dataframe, _ = scaling_and_reshaping_raw_data(test_dataframe)
    X_sequenced, y_sequenced = sequence_scaled_and_reshaped_data(test_dataframe)
    X_train, X_test, y_train, y_test = split_sequenced_data_into_test_and_train(X_sequenced, y_sequenced)
    batched_data = batch_scaled_sequenced_and_reshaped_data(X_train, y_train, 10)
    print("1st dimension of batched data is", len(batched_data))
    print("2nd dimension of batched data is", len(batched_data[0]))
    print("3rd dimension (X_train) of batched data is", len(batched_data[0][0]))
    print("3rd dimension (y_train) of batched data is", len(batched_data[0][1]))
'''
    
def test_331():
    test_LSTM = LSTM()
    #already have default parameter values
    for name, param in test_LSTM.named_parameters():
        print(f"Parameter name: {name}")
        print(f"Shape: {param.shape}")
        print(f"Values: {param.data}\n")
    
def test_332():
    test_LSTM = LSTM()
    X_train, y_train, _, _ = fetched_data_to_sequenced_data_without_scaler("AAPL")
    print(test_LSTM.forward(X_train))
    print(type(test_LSTM.forward(X_train)))
    print(test_LSTM.forward(X_train).shape)
    print("shape of y_train is:", y_train.shape)
    
from LSTM import train_lstm  
    
def test_333():
    X_train, y_train, _, _ = fetched_data_to_sequenced_data_without_scaler("AAPL")
    test_lstm = train_lstm(X_train, y_train)
    #check for epoch and loss prints
    test_prediction = test_lstm.forward(X_train[0].unsqueeze(0))
    #put all but last into forward
    print("prediction is:", test_prediction)
    print("X_train sequence used is:", X_train[0])
    
    
    
    
    

