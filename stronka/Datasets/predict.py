import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


TIME_STEPS = 3
BATCH_SIZE = 32
EPOCHS = 50
LR = 0.001
VAL_SPLIT = 0.1
EARLY_STOP_PATIENCE = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

GRAPHS_DIR = 'graphs'
os.makedirs(GRAPHS_DIR, exist_ok=True)


def clean_column_name(col_name):
    return col_name.replace('/', '_')


# Function to create sequences
def create_sequences(data, time_steps=1, target_idx=0):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:(i + time_steps), :])
        y.append(data[i + time_steps, target_idx])
    return np.array(X), np.array(y)


class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32)


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out


def train_and_evaluate(X_train, y_train, X_val, y_val, input_dim):
    model = LSTMModel(input_dim=input_dim).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    train_dataset = TimeSeriesDataset(X_train, y_train)
    val_dataset = TimeSeriesDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    train_losses = []
    val_losses = []

    best_val_loss = np.inf
    no_improve_count = 0
    best_model_state = None

    for epoch in range(1, EPOCHS + 1):
        # Train
        model.train()
        train_loss_epoch = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs.squeeze(), batch_y)
            loss.backward()
            optimizer.step()

            train_loss_epoch += loss.item() * batch_X.size(0)

        train_loss_epoch /= len(train_loader.dataset)
        train_losses.append(train_loss_epoch)

        # Validate
        model.eval()
        val_loss_epoch = 0
        with torch.no_grad():
            for val_X, val_y in val_loader:
                val_X, val_y = val_X.to(DEVICE), val_y.to(DEVICE)
                val_out = model(val_X)
                v_loss = criterion(val_out.squeeze(), val_y)
                val_loss_epoch += v_loss.item() * val_X.size(0)
        val_loss_epoch /= len(val_loader.dataset)
        val_losses.append(val_loss_epoch)

        print(f"Epoch [{epoch}/{EPOCHS}], Train Loss: {train_loss_epoch:.4f}, Val Loss: {val_loss_epoch:.4f}")

        # Early stopping
        if val_loss_epoch < best_val_loss:
            best_val_loss = val_loss_epoch
            best_model_state = model.state_dict()
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count >= EARLY_STOP_PATIENCE:
                print("Early stopping triggered.")
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, train_losses, val_losses


def predict(model, X_test):
    model.eval()
    with torch.no_grad():
        test_tensor = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
        pred = model(test_tensor)
    return pred.cpu().numpy().flatten()


# Main script: loop over files named "*_Dataset.csv"
for file_path in glob.glob("*_Dataset.csv"):
    city_name = os.path.basename(file_path).split("_")[0]

    # Load data
    df = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
    df = df.sort_index()
    df = df.dropna()

    values = df.values
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(values)

    total_size = len(scaled)
    train_size = int(total_size * 0.8)
    val_size = int(total_size * 0.9)


    predictions_dict = {}
    actual_dict = {}
    test_dates_global = None

    columns = df.columns
    for target_col in columns:
        target_idx = df.columns.get_loc(target_col)

        # Create sequences
        X_all, y_all = create_sequences(scaled, TIME_STEPS, target_idx)

        total_len = len(X_all)
        train_len = int(total_len * 0.8)
        val_len = int(total_len * 0.9)

        X_train, X_val, X_test = X_all[:train_len], X_all[train_len:val_len], X_all[val_len:]
        y_train, y_val, y_test = y_all[:train_len], y_all[train_len:val_len], y_all[val_len:]

        all_dates = df.index[TIME_STEPS:]
        train_dates = all_dates[:train_len]
        val_dates = all_dates[train_len:val_len]
        test_dates = all_dates[val_len:]

        if test_dates_global is None:
            test_dates_global = test_dates

        input_dim = X_all.shape[2]

        print(f"Training model for {city_name} - Target: {target_col}")
        model, train_losses, val_losses = train_and_evaluate(X_train, y_train, X_val, y_val, input_dim)

        y_pred = predict(model, X_test)

        test_start_idx = val_len + TIME_STEPS
        test_scaled_segment = scaled[test_start_idx:test_start_idx + len(y_pred)]

        pred_array = np.copy(test_scaled_segment)
        pred_array[:, target_idx] = y_pred
        inv_pred = scaler.inverse_transform(pred_array)
        inv_y_pred = inv_pred[:, target_idx]

        actual_array = np.copy(test_scaled_segment)
        actual_array[:, target_idx] = y_test
        inv_actual = scaler.inverse_transform(actual_array)
        inv_y_test = inv_actual[:, target_idx]

        clean_col = clean_column_name(target_col)

        predictions_dict[clean_col] = inv_y_pred
        actual_dict[clean_col] = inv_y_test

        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.title(f'{city_name} - {clean_col} Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        loss_plot_path = os.path.join(GRAPHS_DIR, f'{city_name}_{clean_col}_training_loss.png')
        plt.savefig(loss_plot_path, dpi=300)
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.plot(test_dates, inv_y_test, label='Actual', alpha=0.7)
        plt.plot(test_dates, inv_y_pred, label='Predicted', alpha=0.7)
        plt.title(f'{city_name} - {clean_col}: Actual vs Predicted')
        plt.xlabel('Date')
        plt.ylabel(clean_col)
        plt.legend()
        plt.grid(True)
        pred_plot_path = os.path.join(GRAPHS_DIR, f'{city_name}_{clean_col}_actual_vs_predicted.png')
        plt.savefig(pred_plot_path, dpi=300)
        plt.close()

        print(f"Completed {city_name} - {target_col}. Graphs saved in {GRAPHS_DIR}.")

    results_df = pd.DataFrame({'Date': test_dates_global})
    for orig_col in columns:
        c_col = clean_column_name(orig_col)
        results_df[c_col] = predictions_dict[c_col]

    pred_filename = f"{city_name}_predictions.csv"
    results_df.to_csv(pred_filename, index=False)
    print(f"All predictions for {city_name} saved to {pred_filename}.")
