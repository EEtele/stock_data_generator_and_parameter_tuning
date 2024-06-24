from database import get_cassandra_session
import time
from generator import Generator
from gui import plot_price_history_candlesticks, plot_two_price_histories, plot_multiple_price_histories
import winsound
import pandas as pd
from share import Share
import numpy as np
from cassandra.query import BatchStatement, SimpleStatement
from generator_manager import GeneratorManager
from sklearn.preprocessing import MinMaxScaler
from keras.layers import MaxPooling1D
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
from tqdm import tqdm
import uuid
import matplotlib.pyplot as plt

##################CREATE AND GENERATE################################
def create_linked_fBm_shares():
    session, cluster = get_cassandra_session()
    percentage_changes = np.arange(0.1, 2.1, 0.1)  # From 0.1 to 2.0, inclusive
    
    for percent_change in percentage_changes:
        ticker_prefix = f"{int(percent_change * 10):02d}Lin"  # Format: XXLin
        batch_share_details = BatchStatement()
        batch_noise_functions = BatchStatement()
        batch_linkage_functions = BatchStatement()
        
        for i in range(1, 101):  # 100 cases for each percentage_change
            share_ticker = f"{ticker_prefix}{i:02d}"
            batch_share_details.add(
                "INSERT INTO shares.share_details (share_ticker, share_value) VALUES (%s, %s)",
                (share_ticker, 100)
            )
            batch_noise_functions.add(
                "INSERT INTO shares.noise_functions (share_ticker, noise_function_type, noise_parameters) VALUES (%s, %s, %s)",
                (share_ticker, 'FBM', {'hurst_parameter': '0.5'})
            )
            uuid_value = uuid.uuid4()
            batch_linkage_functions.add(
                "INSERT INTO shares.linkage_functions (primary_share_ticker, secondary_share_ticker, linkage_id, linkage_function_type, linkage_parameters) VALUES (%s, %s, %s, %s, %s)",
                (share_ticker, 'XLK', uuid_value, 'percentage_change', {'coefficient': str(percent_change)})
            )

        session.execute(batch_share_details)
        session.execute(batch_noise_functions)
        session.execute(batch_linkage_functions)

    session.shutdown()


def generate_linked_fBm_shares():
    session, cluster = get_cassandra_session()
    coefficient_ranges = np.arange(0.1, 2.1, 0.1)
    share_tickers = [f"{int(coef * 10):02d}Lin{i:02d}" for coef in coefficient_ranges for i in range(1, 101)]
    generator_manager = GeneratorManager(share_tickers, session)
    generator_manager.generate()
    session.shutdown()
#####################################################################


#################SAVE AND LOAD#######################################
def fBm_export_main():
    session, cluster = get_cassandra_session()
    coefficient_ranges = np.arange(0.1, 2.1, 0.1)
    df_all = pd.DataFrame()

    # Generate all share tickers based on the coefficient ranges
    share_tickers = [f"{int(coef * 10):02d}Lin{i:02d}" for coef in coefficient_ranges for i in range(1, 101)]

    data_list = []
    coefficient_list = []

    for ticker in share_tickers:
        share = Share(ticker, session)
        result = share.get_all_values()
        df = pd.DataFrame(list(result))
        
        coefficient = float(ticker[:2]) / 10.0  # Convert '10Lin67' back to 1.0

        data_series = df['value'].values
        data_list.append(data_series)
        coefficient_list.append(coefficient)

    session.shutdown()

    return np.array(data_list), np.array(coefficient_list)

def save_to_csv(data, coefficients, filename):
    data_strings = [' '.join(map(str, series)) for series in data]
    df = pd.DataFrame({'data': data_strings, 'coefficient': coefficients})
    df.to_csv(filename, index=False)

def load_from_csv_plus(filename = 'link_data.csv'):
    session, cluster = get_cassandra_session()
    query = "SELECT * FROM shares.price_history WHERE share_ticker = %s"
    result_set = session.execute(query, ("XLK",))
    session.shutdown()
    df = pd.read_csv(filename)
    data_list = np.array([np.array(list(map(float, series.split()))) for series in df['data']])
    coefficient_list = np.array(df['coefficient'].values)
    related_array = np.array([row[2] for row in result_set])
    related_array = related_array.reshape(1, -1)
    related_array_repeated = np.repeat(related_array, data_list.shape[0], axis=0)
    combined_data = np.concatenate((data_list, related_array_repeated), axis=1)
    reshaped_data_list = combined_data.reshape(data_list.shape[0], 2, -1)

    return reshaped_data_list, coefficient_list

#####################################################################


#################TRAIN AND PREDICT###################################

class LSTMModel(nn.Module):
    def __init__(self):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=1461, hidden_size=128, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(128, 1)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        x = self.fc1(h_n.squeeze(0))
        return x



class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.fc1 = nn.Linear(32 * 730, 1)  # Adjusted based on the output of the last pool layer

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

class CNNModel2(nn.Module): #okay
    def __init__(self):
        super(CNNModel2, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 2))

        self.conv2 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 2))

        self.conv3 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=(1, 2))

        self.fc1 = nn.Linear(256 * 2 * 182, 512) 
        self.drop = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1) 
        x = self.drop(self.fc1(x))
        x = self.fc2(x)
        return x


class RevisedCNNModel(nn.Module):
    def __init__(self):
        super(RevisedCNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 2))

        self.conv2 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 2))

        self.conv4 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.relu4 = nn.ReLU()

        self.conv5 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.relu5 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=(1, 2))

        self.conv6 = nn.Conv2d(256, 512, kernel_size=(3, 3), stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(512)
        self.relu6 = nn.ReLU()

        self.conv7 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=1, padding=1)
        self.bn7 = nn.BatchNorm2d(512)
        self.relu7 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=(1, 2))

        # Correct the input dimension based on the flattened output from the last pooling layer
        self.fc1 = nn.Linear(512 * 2 * 91, 512)  # Adjusted
        self.drop = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.pool2(self.relu3(self.bn3(self.conv3(x))))
        x = self.relu4(self.bn4(self.conv4(x)))
        x = self.pool3(self.relu5(self.bn5(self.conv5(x))))
        x = self.relu6(self.bn6(self.conv6(x)))
        x = self.pool4(self.relu7(self.bn7(self.conv7(x))))
        x = x.view(x.size(0), -1)  # Flatten the output
        x = self.drop(self.fc1(x))
        x = self.fc2(x)
        return x
    


def train_model(model, train_loader, val_loader, epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    model.train()

    # Prepare to track loss for plots
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            progress_bar.set_postfix({'Train Loss': loss.item()})

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation loss calculation
        val_loss = evaluate_model(model, val_loader, training=True)
        val_losses.append(val_loss)
        
        print(f'Epoch {epoch+1}, Loss: {avg_train_loss}, Val Loss: {val_loss}')

    print("Training complete.")

    # Plot losses
    plt.figure(figsize=(10, 5))
    #make max y value 50
    plt.ylim(0, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss Trajectory')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    return train_losses, val_losses


def evaluate_model(model, data_loader, training=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    criterion = nn.MSELoss()
    total_loss = 0
    predictions, actuals = [], []

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets)
            total_loss += loss.item()
            predictions.extend(outputs.squeeze().cpu().tolist())
            actuals.extend(targets.cpu().tolist())

    avg_loss = total_loss / len(data_loader)
    errors = [pred - actual for pred, actual in zip(predictions, actuals)]
    avg_error = np.mean(np.abs(errors))
    if not training:
        print(f'Average Loss: {avg_loss}')
        print(f'Average Error: {avg_error}')

        # Plotting average errors
        error_dict = {}
        for act, err in zip(actuals, errors):
            if act not in error_dict:
                error_dict[act] = []
            error_dict[act].append(err)

        avg_pos_errors = []
        avg_neg_errors = []
        grouped_actuals = sorted(list(set(actuals)))
        for act in grouped_actuals:
            pos_errors = [e for e in error_dict[act] if e > 0]
            neg_errors = [e for e in error_dict[act] if e < 0]
            avg_pos_errors.append(np.mean(pos_errors) if pos_errors else 0)
            avg_neg_errors.append(np.mean(neg_errors) if neg_errors else 0)

        fig, ax = plt.subplots()
        ax.bar(grouped_actuals, avg_pos_errors, width=0.05, label='Average Positive Errors', color='blue', alpha=0.7)
        ax.bar(grouped_actuals, avg_neg_errors, width=0.05, label='Average Negative Errors', color='red', alpha=0.7)
        ax.set_xlabel('Actual Values')
        ax.set_ylabel('Average Errors')
        ax.set_title('Average Positive and Negative Errors by Actual Value')
        ax.legend()
        plt.show()

    return avg_loss

#####################################################################


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    # Load data, shape should be  similar to (2000, 2, 1461), (2000,), with only 2000 being variable.
    data, coefficients = load_from_csv_plus()
    
    print("Loaded data shape:", data.shape)
    print("Loaded coefficients shape:", coefficients.shape)
    
    # Prepare the data tensor
    data_tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(1)  # Add channel dimension for CNN
    coeff_tensor = torch.tensor(coefficients, dtype=torch.float32)

    print("Data tensor shape:", data_tensor.shape)
    print("Coefficients tensor shape:", coeff_tensor.shape)

    # Create dataset and dataloader
    dataset = TensorDataset(data_tensor, coeff_tensor)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    # Initialize the model
    model = RevisedCNNModel().to(device)
    print("Model initialized.")

    # Train the model
    train_model(model, train_loader, val_loader, epochs=25)

    # Evaluate the model on the validation set for performance assessment
    evaluate_model(model, val_loader)

    #torch.save(model.state_dict(), 'model3.pth')
    #print("Model saved to model3.pth")


if __name__ == "__main__":
    main()

