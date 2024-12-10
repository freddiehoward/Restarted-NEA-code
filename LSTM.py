import torch
from torch import nn
from data_preprocessing import fetched_data_to_sequenced_data_without_scaler

class LSTM(nn.Module):
    """
    LSTM model for time-series prediction.
    """
    def __init__(self, input_size=1, hidden_size=50, output_size=1, num_layers=1):
        
        super(LSTM, self).__init__()
        #super is initialising the nn.Module class as this parent class sets up all
        #the parameters and allows us to switch between training and eval mode etc. for our model
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Forward pass for the LSTM model.

        Inputs:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_size), is a singular batch

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1, output_size). 1 output per sequence in the batch
        """
        
        
        out, _ = self.lstm(x)
        #self.lstm(x) runs the forward function defined in nn.Module but it is clearer if I effectively
        #define it/call it again in the LSTM class
        #out is the hidden state of each timestep in each sequence in a batch
        #out should have shape (num_sequences, sequence_length, hidden_size) = (len(batched_data[0]), 50, 50)
        #btw i arbitrarily decided that 50 would be a suitable length for sequences, given that I intend to use 1 year of data
        #meaning roughly 250 days of data (no weekends or bank holidays)
        #the second output of self.lstm(x) is the hidden state and cell state of the last timestep of each sequence in a batch
        #which I don't need for now, so I don't store it in a variable
        
        #since we only want the final hidden state value of each sequence, not the hidden state values for each timestep
        #the code below achieves this, with -1 meaning the final value
        out = self.fc(out[:, -1, :])
        
        #ie the shape of out from the model should be (num_of_sequences, length_of_output_sequence, output_size)(num_of_sequences, 1, 1)
        #this is because for each sequence we want a prediction one day into the future, hence the first and second dimension,
        #and we only want the close price in the prediction hence the third dimension
        
        #may need to reshape the output
        out = out.reshape(-1,1,1)
        
        return out
    


def train_lstm(X_train, y_train, input_size=1, hidden_size=50, output_size=1, num_epochs=200):
    """
    Train an LSTM model on provided training data.

    Inputs:
        X_train (torch.Tensor): All Training input data ie unbatched of shape (num_of_sequences, seq_len, input_size).
        y_train (torch.Tensor): Training target data unbatched of shape (num_of_sequences, output_size). ie 1 output per sequence
        input_size (int): Number of input features (default is 1).
        hidden_size (int): Number of hidden units in the LSTM (default is 50).
        output_size (int): Number of output features (default is 1).
        num_epochs (int): Number of epochs to train the model (default is 20).

    Returns:
        torch.nn.Module: Trained LSTM model.
    """
    
    # Initialize the model
    model = LSTM(input_size, hidden_size, output_size)
    
    #loss function
    criterion = nn.MSELoss()
    
    #optimiser to perform backprop
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(num_epochs):
        #epoch is when all the training data is used for training ie all batches trained with
        
        model.train()
        #puts the model into training mode, this doesn't train it though.
        
        epoch_loss = 0.0
        #resets the loss after each epoch otherwise the loss would accumulate every epoch so logs would be incorrect
    
        #Zero the gradients
        optimizer.zero_grad()
        outputs = model(X_train)
        
        #squeeze removes dimesnions with a value of 1 ie outputs.shape is (batch_size, 1, hidden_size),
        #after squeeze it is (batch_size, hidden_size), which matches y_train.shape so we can calculate
        #the loss
        loss = criterion(outputs, y_train)
        
        #Computes the gradients of the loss with respect to all model parameters.
        loss.backward()
        
        #changes the model parameters using the calculated gradients above
        optimizer.step()
        epoch_loss += loss.item()

        # Log the loss for the epoch
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    return model
