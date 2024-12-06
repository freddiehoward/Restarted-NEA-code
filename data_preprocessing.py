
from sklearn.preprocessing import MinMaxScaler


def scaling_and_reshaping_raw_data(data):
    
    '''
    Inputs:
        data (pandas.DataFrame): Stock data with a 'Close' column. Unsequenced
        
    Outputs:
        scaled_data (pandas.DataFrame): Scaled Stock data with a 'Close' column. Unsequenced
    '''
    
    
    # Scale the 'Close' prices to a range between 0 and 1, WHY?
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    #fit_transform?
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
    #scaled data is now 2D
    
    return scaled_data, scaler
    


import numpy as np
from sklearn.model_selection import train_test_split

#MAY BE OK TO REMOVE
#import pandas as pd

def sequence_scaled_and_reshaped_data(scaled_data, scaler, sequence_length=60):
    """
    Preprocess stock data for LSTM model training.

    Inputs:
        scaled_data (pandas.DataFrame): Scaled Stock data with a 'Close' column. Unsequenced. shape(num_data_points, 1)
        sequence_length (int): Number of time steps we want in each sequence.

    Returns:
        3D and 2D arrays: Scaled training and testing data (X_train, X_test, y_train, y_test).
    """
    
    #eg scaled data = [[0.45], [0.23], [0.11], [-0.19], [-0.13], [-0.15]]
    #eg seq_length = 2
    #eg X_sequenced = [[[0.45], [0.23]], [[0.23], [0.11]], [[0.11], [-0.19]], [[-0.19], [-0.13]], [[-0.13], [-0.15]]]
    #eg y_sequenced = [[0.11], [-0.19], [-0.13], [-0.15]]

    # Create sequences of data
    X_sequenced, y_sequenced = [], []
    for i in range(sequence_length, len(scaled_data)):
        X_sequenced.append(scaled_data[i-sequence_length:i, 0])
        y_sequenced.append(scaled_data[i, 0])

    X_sequenced, y_sequenced = np.array(X_sequenced), np.array(y_sequenced)
    X_sequenced = np.reshape(X_sequenced, (X_sequenced.shape[0], X_sequenced.shape[1], 1))  # Reshape for LSTM input

    # Split into training and testing sets randomly, degree of randomness is random_state value
    X_train, X_test, y_train, y_test = train_test_split(X_sequenced, y_sequenced, test_size=0.2, random_state=42)
    

    return X_train, X_test, y_train, y_test, scaler


def batch_scaled_sequenced_and_reshaped_data(X_sequenced, y, batch_size):
    
    #X_sequenced and y_sequenced must have already been split into sequences
    
    """
    Create batches of data manually.

    Inputs:
        X_sequenced (torch.Tensor): Input data of shape (num_of_sequences, seq_len, input_size).
        y_sequenced (torch.Tensor): Target data of shape (num_of_sequences, output_size) as 1 output per sequence and output_size is 1
        batch_size (int): Size of each batch.

    Returns:
        list: List of arrays, where each array contains a batch of [X_batch, y_batch].
    """
    n_sequences = X.shape[0]
    batches = []
    
    for i in range(0, n_sequences, batch_size):
        
        
        X_batch = X_sequenced[i:i + batch_size]
        
        y_batch = y[i:i + batch_size]
        
        batches.append([X_batch, y_batch])
        
    return batches



