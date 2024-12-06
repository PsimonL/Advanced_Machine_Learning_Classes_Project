import torch
import torch.nn as nn
import joblib


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(LSTMModel, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Define LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)

        # Define a fully connected layer
        self.fc = nn.Linear(hidden_size, output_size)  # Output is a sequence of length forecast_horizon

    def forward(self, x):
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: (batch, seq_length, hidden_size)

        # Take the output from the last time step
        out = out[:, -1, :]  # (batch, hidden_size)

        # Pass through the fully connected layer
        out = self.fc(out)  # (batch, output_size)
        return out  # (batch, forecast_horizon)


# Load the model
input_size = 1
hidden_size = 16
num_layers = 2
output_size = 1
dropout = 0.2

model = LSTMModel(input_size, hidden_size, num_layers, output_size, dropout)
model.load_state_dict(torch.load('./stronka/PT/O3_PT_lstm_model.pth', map_location=torch.device('cpu')))
model.eval()

scaler = joblib.load('./stronka/PT/scaler.save')
