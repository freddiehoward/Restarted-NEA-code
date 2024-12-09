
from sklearn.preprocessing import MinMaxScaler


def scaling_and_reshaping_raw_data(data):
    
    '''
    Inputs:
        data (pandas.DataFrame): Stock data with a 'Close' column. Unsequenced
        
    Outputs:
        scaled_data (pandas.DataFrame): Scaled Stock data with a 'Close' column. Unsequenced
    '''
    
    
    # Scale the 'Close' prices to a range between 0 and 1, to keep them within sensitive ranges for activation functions
    #which aren't sensitive to changes at extreme values eg the sigmoid function
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    #fit_transform?
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
    #scaled data is now 2D
    
    return scaled_data, scaler
    

import numpy as np
from sklearn.model_selection import train_test_split

def sequence_scaled_and_reshaped_data(scaled_data, sequence_length=60):
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
    y_sequenced = np.reshape(y_sequenced, (y_sequenced.shape[0], 1, 1))  # Reshape expected output

    # Split into training and testing sets randomly, degree of randomness is random_state value
    
    return X_sequenced, y_sequenced
    

def split_sequenced_data_into_test_and_train(X_sequenced, y_sequenced):
    """
    Split the sequenced data into training and testing sets.
    
    Inputs:
        X_sequenced (numpy.ndarray): Sequenced input data. shape(num_data_points, sequence_length, input_size)
        y_sequenced (numpy.ndarray): Target data. shape(num_data_points, sequence_length, output_size)
        
    Returns:
        3D and 2D arrays: Sequenced training and testing data (X_train, X_test, y_train, y_test).
    
    """
    X_train, X_test, y_train, y_test = train_test_split(X_sequenced, y_sequenced, test_size=0.2, random_state=42)
    

    return X_train, X_test, y_train, y_test
    


def batch_scaled_sequenced_and_reshaped_data(X_sequenced, y_sequenced, num_of_sequences):
    
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
    n_sequences = X_sequenced.shape[0]
    batches = []
    
    for i in range(0, n_sequences, num_of_sequences):
        
        
        X_batch = X_sequenced[i:i + num_of_sequences]
        
        y_batch = y_sequenced[i:i + num_of_sequences]
        
        batches.append([X_batch, y_batch])
        
    return batches



