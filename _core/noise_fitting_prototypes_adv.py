from database import get_cassandra_session
import time
from generator import Generator
from gui import plot_price_history_candlesticks, plot_two_price_histories, plot_multiple_price_histories
import winsound
import pandas as pd
from share import Share
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
from cassandra.query import BatchStatement, SimpleStatement
from generator_manager import GeneratorManager
from sklearn.preprocessing import MinMaxScaler
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import matplotlib.pyplot as plt

##################CREATE AND GENERATE################################
def generate_fBm_shares():
    session, cluster = get_cassandra_session()
    share_tickers = []
    hurst_values = np.arange(0.10, 0.92, 0.02)
    for hurst in hurst_values:
        for i in range(1, 101):
            share_tickers.append(f"{hurst:.2f}".replace('.', '') + f"fBm{i}")
    generator_manager = GeneratorManager(share_tickers, session)
    generator_manager.generate()
    session.shutdown()  # Added this to close the session after generating

def create_fBm_shares_batch():
    session, cluster = get_cassandra_session()
    hurst_values = np.arange(0.10, 0.92, 0.02)
    
    for hurst in hurst_values:
        batch_share_details = BatchStatement()
        batch_noise_functions = BatchStatement()
        
        for i in range(1, 101):
            share_ticker = f"{hurst:.2f}".replace('.', '') + f"fBm{i}"
            batch_share_details.add(
                "INSERT INTO shares.share_details (share_ticker, share_value) VALUES (%s, %s)",
                (share_ticker, 100)
            )
            batch_noise_functions.add(
                "INSERT INTO shares.noise_functions (share_ticker, noise_function_type, noise_parameters) VALUES (%s, %s, %s)",
                (share_ticker, 'FBM', {'hurst_parameter': str(hurst)})
            )
        
        session.execute(batch_share_details)
        session.execute(batch_noise_functions)
    
    session.shutdown()
#####################################################################


#################SAVE AND LOAD#######################################
def fBm_export_main():
    session, cluster = get_cassandra_session()
    hurst_values = np.arange(0.10, 0.92, 0.02)
    df_all = pd.DataFrame()

    # Generate all share tickers
    share_tickers = [f"{hurst:.2f}".replace('.', '') + f"fBm{i}" for hurst in hurst_values for i in range(1, 101)]

    # List to hold data and corresponding Hurst parameters
    data_list = []
    hurst_params_list = []

    # Loop over each ticker, fetch data, and append to the main DataFrame
    for ticker in share_tickers:
        share = Share(ticker, session)
        result = share.get_all_values()
        df = pd.DataFrame(list(result))
        
        # Extract the Hurst parameter from the ticker name
        hurst_param = float(ticker[:3]) / 100.0  # Convert '010fBm67' to 0.10, etc.

        # Assume 'value' column contains the time series data
        data_series = df['value'].values
        data_list.append(data_series)
        hurst_params_list.append(hurst_param)

    session.shutdown()

    return np.array(data_list), np.array(hurst_params_list)

def save_to_csv(data, hurst_params, filename):
    # Create a list of space-separated string representations of the data series
    data_strings = [' '.join(map(str, series)) for series in data]
    
    # Create a DataFrame with the data strings and Hurst parameters
    df = pd.DataFrame({'data': data_strings, 'hurst_param': hurst_params})
    
    # Save the DataFrame to a CSV file
    df.to_csv(filename, index=False)



def load_from_csv(filename):
    # Load the CSV file into a DataFrame
    print
    df = pd.read_csv(filename)
    
    # Convert the space-separated strings back to arrays of floats
    data_list = [np.array(list(map(float, series.split()))) for series in df['data']]
    
    # Extract the Hurst parameters
    hurst_params_list = df['hurst_param'].values
    
    return np.array(data_list), np.array(hurst_params_list)

#####################################################################


#################TRAIN AND PREDICT###################################




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def neural_model_focus_():
    data, hurst_params = load_from_csv("fBm_data.csv")
    scaler = MinMaxScaler()

    data = np.array(data).astype(float)
    data = scaler.fit_transform(data.reshape(-1, 1)).reshape(data.shape)
    data = data.reshape((data.shape[0], data.shape[1], 1))
    data = np.transpose(data, (0, 2, 1))
    print(data.shape)

    X = torch.tensor(data, dtype=torch.float32).to(device)
    y = torch.tensor(hurst_params, dtype=torch.float32).to(device)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=16, shuffle=False)

    class CNNModel(nn.Module):
        def __init__(self):
            super(CNNModel, self).__init__()
            self.cnn_layers = nn.Sequential(
                nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2),
                nn.Dropout(0.3),
                nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Flatten(),
                nn.Linear(128 * ((1461 // 2)), 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )

        def forward(self, x):
            return self.cnn_layers(x).squeeze(-1)

    model = CNNModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # Tracking training and validation loss
    train_losses = []
    val_losses = []

    for epoch in range(100):
        model.train()
        train_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/100", unit="batch"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_loss /= len(test_loader)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}: Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

    # Plotting the training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

def neural_model_focus():
    data, hurst_params = load_from_csv("fBm_data.csv")
    scaler = MinMaxScaler()

    data = np.array(data).astype(float)
    data = scaler.fit_transform(data.reshape(-1, 1)).reshape(-1, 1461)
    data = data.reshape((data.shape[0], data.shape[1], 1))
    data = np.transpose(data, (0, 2, 1))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    X = torch.tensor(data, dtype=torch.float32).to(device)
    y = torch.tensor(hurst_params, dtype=torch.float32).to(device)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=16, shuffle=False)

    class CNNModel(nn.Module):
        def __init__(self):
            super(CNNModel, self).__init__()
            self.cnn_layers = nn.Sequential(
                nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2),
                nn.Dropout(0.3),
                nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Flatten(),
                nn.Linear(128 * (1461 // 2), 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )

        def forward(self, x):
            return self.cnn_layers(x).squeeze(-1)

    model = CNNModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    train_losses = []
    val_losses = []

    # Training loop
    for epoch in range(100):
        model.train()
        train_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/100", unit="batch"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_loss /= len(test_loader)
        val_losses.append(val_loss)

    # Final evaluation step
    final_real_params = []
    final_predicted_params = []
    model.eval()
    with torch.no_grad():
        final_loss = 0.0
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            final_loss += loss.item()
            final_real_params.extend(labels.tolist())
            final_predicted_params.extend(outputs.tolist())
    final_loss /= len(test_loader)

    avg_difference = np.mean(np.abs(np.array(final_real_params) - np.array(final_predicted_params)))

    # Plotting results
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    print(f"Final Test Loss: {final_loss:.4f}")
    print("Real parameters:", final_real_params)
    print("Predicted parameters:", final_predicted_params)
    print("Average difference between real and predicted parameters:", avg_difference)


#####################################################################
if __name__ == "__main__":
    neural_model_focus()