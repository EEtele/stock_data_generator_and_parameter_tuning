from database import get_cassandra_session
from cassandra.query import BatchStatement
import time
from generator import Generator
from generator_manager import GeneratorManager
from gui import plot_price_history_candlesticks, plot_two_price_histories, plot_multiple_price_histories
import winsound
import pandas as pd
from share import Share
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
from tqdm import tqdm
import uuid
import matplotlib.pyplot as plt



def create_unified_shares():
    session, cluster = get_cassandra_session()
    hurst_values = np.arange(0.00, 0.92, 0.02)  # Starting from 0.00 now
    percentage_changes = np.arange(-3.0, 3.1, 0.1)  # Ranging from -3 to +3
    jj = 1

    for hurst in hurst_values:
        for percent_change in percentage_changes:
            hurst_part = f"{hurst:.2f}".replace('.', '')
            if percent_change < 0:
                percent_part = f"N{int(abs(percent_change) * 10):02d}"
            else:
                percent_part = f"{int(percent_change * 10):02d}"

            batch_share_details = BatchStatement()
            batch_noise_functions = BatchStatement()
            batch_linkage_functions = BatchStatement()
            
            for i in range(1, 5):  # Generating 4 shares per combination
                share_ticker = f"{hurst_part}{percent_part}Uni{i:02d}"
                print(f"Creating share {share_ticker}   nr:", jj)
                jj += 1
                batch_share_details.add(
                    "INSERT INTO shares.share_details (share_ticker, share_value) VALUES (%s, %s)",
                    (share_ticker, 100)
                )
                batch_noise_functions.add(
                    "INSERT INTO shares.noise_functions (share_ticker, noise_function_type, noise_parameters) VALUES (%s, %s, %s)",
                    (share_ticker, 'FBM', {'hurst_parameter': str(hurst)})
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

def generate_unified_fBm_shares():
    session, cluster = get_cassandra_session()
    hurst_values = np.arange(0.00, 0.92, 0.02)
    coefficient_ranges = np.arange(-3.0, 3.1, 0.1)
    batch_size = 50  # Adjust batch size based on your memory and performance requirements

    share_tickers = [
        f"{hurst:.2f}".replace('.', '') + (f"N{int(abs(coef) * 10):02d}" if coef < 0 else f"{int(coef * 10):02d}") + f"Uni{i:02d}"
        for hurst in hurst_values
        for coef in coefficient_ranges
        for i in range(1, 5)
    ]

    for i in range(0, len(share_tickers), batch_size):
        batch_tickers = share_tickers[i:i+batch_size]
        generator_manager = GeneratorManager(batch_tickers, session)
        generator_manager.generate()

    session.shutdown()

def unified_fBm_export_main():
    session, cluster = get_cassandra_session()
    hurst_values = np.arange(0.00, 0.92, 0.02)
    coefficient_ranges = np.arange(-3.0, 3.1, 0.1)
    batch_size = 50  # Same batch size as in generation

    share_tickers = [
        f"{hurst:.2f}".replace('.', '') + (f"N{int(abs(coef) * 10):02d}" if coef < 0 else f"{int(coef * 10):02d}") + f"Uni{i:02d}"
        for hurst in hurst_values
        for coef in coefficient_ranges
        for i in range(1, 5)
    ]

    data_list = []
    hurst_params_list = []
    coefficient_list = []

    for i in range(0, len(share_tickers), batch_size):
        batch_tickers = share_tickers[i:i+batch_size]
        for ticker in batch_tickers:
            share = Share(ticker, session)
            result = share.get_all_values()
            df = pd.DataFrame(list(result))

            hurst_param = float(ticker[:3]) / 100.0
            coeff_str = ticker[3:5]
            coefficient = -float(coeff_str[1:]) / 10.0 if 'N' in coeff_str else float(coeff_str) / 10.0
            data_series = df['value'].values

            data_list.append(data_series)
            hurst_params_list.append(hurst_param)
            coefficient_list.append(coefficient)

    session.shutdown()
    return np.array(data_list), np.array(hurst_params_list), np.array(coefficient_list)

def save_to_unified_csv(data, hurst_params, coefficients, filename='extended_range_unified.csv'):
    data_strings = [' '.join(map(str, series)) for series in data]
    df = pd.DataFrame({
        'data': data_strings,
        'hurst_param': hurst_params,
        'coefficient': coefficients
    })
    df.to_csv(filename, index=False)

def uniform_downsample(target_length, data_list, hurst_params, coefficients):
    current_length = len(data_list)
    if current_length <= target_length:
        return np.array(data_list), np.array(hurst_params), np.array(coefficients)

    # Generating indices using linspace to ensure they are evenly distributed
    indices = np.linspace(0, current_length - 1, num=target_length, dtype=int)
    
    downsampled_data_list = [data_list[i] for i in indices]
    downsampled_hurst_params = hurst_params[indices]
    downsampled_coefficients = coefficients[indices]

    return np.array(downsampled_data_list), np.array(downsampled_hurst_params), np.array(downsampled_coefficients)

def load_from_csv(filename='../data/extended_range_unified.csv', coef_range=(-5, 5), hurst_range=(0, 1.1)):
    df = pd.read_csv(filename)
    # Filter the dataframe based on the coefficient and Hurst parameter ranges
    filtered_df = df[(df['coefficient'] >= coef_range[0]) & (df['coefficient'] <= coef_range[1]) &
                     (df['hurst_param'] >= hurst_range[0]) & (df['hurst_param'] <= hurst_range[1])]
    
    data_list = [np.array(list(map(float, series.split()))) for series in filtered_df['data']]
    hurst_params_list = filtered_df['hurst_param'].values
    coefficient_list = filtered_df['coefficient'].values
    
    return np.array(data_list), np.array(hurst_params_list), np.array(coefficient_list)

def load_from_csv_plus(filename='../data/extended_range_unified.csv', coef_range=(-5, 5), hurst_range=(0, 1.1)):
    session, cluster = get_cassandra_session()
    query = "SELECT * FROM shares.price_history WHERE share_ticker = %s"
    result_set = session.execute(query, ("XLK",))
    session.shutdown()
    
    df = pd.read_csv(filename)
    # Filter the dataframe based on the coefficient
    filtered_df = df[(df['coefficient'] >= coef_range[0]) & (df['coefficient'] <= coef_range[1]) &
                     (df['hurst_param'] >= hurst_range[0]) & (df['hurst_param'] <= hurst_range[1])]
    
    data_list = np.array([np.array(list(map(float, series.split()))) for series in filtered_df['data']])
    coefficient_list = np.array(filtered_df['coefficient'].values)
    hurst_params_list = np.array(filtered_df['hurst_param'].values)
    
    related_array = np.array([row[2] for row in result_set]) 
    related_array = related_array.reshape(1, -1)
    related_array_repeated = np.repeat(related_array, data_list.shape[0], axis=0)
    combined_data = np.concatenate((data_list, related_array_repeated), axis=1)
    reshaped_data_list = combined_data.reshape(data_list.shape[0], 2, -1)

    return reshaped_data_list, hurst_params_list, coefficient_list

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

        self.fc1 = nn.Linear(512 * 2 * 91, 512)
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
        x = x.view(x.size(0), -1)
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

    # # Plot losses
    # plt.figure(figsize=(10, 5))
    # #make max y value 50
    # plt.ylim(0, 1)
    # plt.plot(train_losses, label='Train Loss')
    # plt.plot(val_losses, label='Validation Loss')
    # plt.title('Loss Trajectory')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.legend()
    # plt.show()

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
        # error_dict = {}
        # for act, err in zip(actuals, errors):
        #     if act not in error_dict:
        #         error_dict[act] = []
        #     error_dict[act].append(err)

        # avg_pos_errors = []
        # avg_neg_errors = []
        # grouped_actuals = sorted(list(set(actuals)))
        # for act in grouped_actuals:
        #     pos_errors = [e for e in error_dict[act] if e > 0]
        #     neg_errors = [e for e in error_dict[act] if e < 0]
        #     avg_pos_errors.append(np.mean(pos_errors) if pos_errors else 0)
        #     avg_neg_errors.append(np.mean(neg_errors) if neg_errors else 0)

        # fig, ax = plt.subplots()
        # ax.bar(grouped_actuals, avg_pos_errors, width=0.05, label='Average Positive Errors', color='blue', alpha=0.7)
        # ax.bar(grouped_actuals, avg_neg_errors, width=0.05, label='Average Negative Errors', color='red', alpha=0.7)
        # ax.set_xlabel('Actual Values')
        # ax.set_ylabel('Average Errors')
        # ax.set_title('Average Positive and Negative Errors by Actual Value')
        # ax.legend()
        # plt.show()

    return avg_loss

def create_tensor_dataset(data, coefficients):
    data_tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(1)
    coeff_tensor = torch.tensor(coefficients, dtype=torch.float32)
    return TensorDataset(data_tensor, coeff_tensor)

def create_data_loaders_(dataset):
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    return train_loader, val_loader

def create_data_loaders(dataset, seed=42):
    # Set the seed for generating random numbers
    torch.manual_seed(seed)
    
    # You might also want to ensure reproducibility by doing the following:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    return train_loader, val_loader

def comparative_evaluation(model1, model2, data_loader1, data_loader2):
    print("Evaluating both models on both datasets:")
    # Collect errors and actuals for detailed analysis and comparison
    errors1_dataset1, actuals1_dataset1 = evaluate_model_(model1, data_loader1)
    errors1_dataset2, actuals1_dataset2 = evaluate_model_(model1, data_loader2)
    errors2_dataset1, actuals2_dataset1 = evaluate_model_(model2, data_loader1)
    errors2_dataset2, actuals2_dataset2 = evaluate_model_(model2, data_loader2)

    # Plotting comparisons
    print("Comparing Model 1 and Model 2 on Dataset 1")
    analyze_errors(errors1_dataset1, actuals1_dataset1, errors2_dataset1, actuals2_dataset1, "Dataset 1")

    print("Comparing Model 1 and Model 2 on Dataset 2")
    analyze_errors(errors1_dataset2, actuals1_dataset2, errors2_dataset2, actuals2_dataset2, "Dataset 2")

def evaluate_model_(model, data_loader, training=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.MSELoss()
    total_loss = 0
    predictions, actuals = [], []

    model.eval()
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets)
            total_loss += loss.item()
            predictions.extend(outputs.squeeze().cpu().tolist())
            actuals.extend(targets.cpu().tolist())

    avg_loss = total_loss / len(data_loader)
    print(f'Average Loss: {avg_loss}')
    errors = [pred - actual for pred, actual in zip(predictions, actuals)]
    return errors, actuals

def analyze_errors_(errors1, actuals1, errors2, actuals2, dataset_label):
    error_dict1 = group_errors_by_actual(errors1, actuals1)
    error_dict2 = group_errors_by_actual(errors2, actuals2)

    fig, ax = plt.subplots(2, 1, figsize=(12, 10), tight_layout=True)

    # Plot average errors by actual values for both models
    plot_average_errors(ax[0], error_dict1, 'Model 1', 'blue',-0.1)
    plot_average_errors(ax[0], error_dict2, 'Model 2', 'red', 0.1)
    ax[0].set_title(f'Average Positive and Negative Errors by Actual Value on {dataset_label}')
    ax[0].legend()

    # Histogram of all errors combined
    ax[1].hist(errors1, bins=50, color='blue', alpha=0.7, label='Model 1 Errors')
    ax[1].hist(errors2, bins=50, color='red', alpha=0.7, label='Model 2 Errors')
    ax[1].set_xlabel('Error')
    ax[1].set_ylabel('Frequency')
    ax[1].set_title(f'Histogram of Prediction Errors on {dataset_label}')
    ax[1].legend()

    plt.show()

def plot_average_absolute_errors(ax, error_dict, label, color):
    actuals = sorted(error_dict.keys())
    avg_abs_errors = [sum(abs(e) for e in error_dict[actual]) / len(error_dict[actual]) for actual in actuals]
    ax.plot(actuals, avg_abs_errors, label=label, color=color, marker='o', linestyle='-')

def analyze_errors(errors1, actuals1, errors2, actuals2, dataset_label):
    error_dict1 = group_errors_by_actual(errors1, actuals1)
    error_dict2 = group_errors_by_actual(errors2, actuals2)

    fig, ax = plt.subplots(3, 1, figsize=(12, 15), tight_layout=True)

    # Plot average errors by actual values for both models
    plot_average_errors(ax[0], error_dict1, 'Model 1', 'blue', -0.1)
    plot_average_errors(ax[0], error_dict2, 'Model 2', 'red', 0.1) 
    ax[0].set_title(f'Average Positive and Negative Errors by Actual Value on {dataset_label}')
    ax[0].legend()

    # Histogram of all errors combined
    ax[1].hist(errors1, bins=50, color='blue', alpha=0.7, label='Model 1 Errors')
    ax[1].hist(errors2, bins=50, color='red', alpha=0.7, label='Model 2 Errors')
    ax[1].set_xlabel('Error')
    ax[1].set_ylabel('Frequency')
    ax[1].set_title(f'Histogram of Prediction Errors on {dataset_label}')
    ax[1].legend()

    # Plot average absolute errors as lines
    plot_average_absolute_errors(ax[2], error_dict1, 'Model 1 Absolute Errors', 'blue')
    plot_average_absolute_errors(ax[2], error_dict2, 'Model 2 Absolute Errors', 'red')
    ax[2].set_xlabel('Actual Value')
    ax[2].set_ylabel('Average Absolute Error')
    ax[2].set_title('Average Absolute Errors by Actual Value')
    ax[2].legend()

    plt.show()

def plot_average_errors(ax, error_dict, label, color, offset):
    avg_pos_errors, avg_neg_errors, grouped_actuals = calculate_average_errors(error_dict)
    width = 0.2
    indices = np.arange(len(grouped_actuals))
    ax.bar(indices + offset, avg_pos_errors, width=width, label=f'{label} Average Positive Errors', color=color, alpha=0.7)
    ax.bar(indices + offset, avg_neg_errors, width=width, bottom=0, label=f'{label} Average Negative Errors', color=color, edgecolor='black', alpha=0.7)

    ax.set_xticks(indices)
    ax.set_xticklabels([f'{x:.2f}' for x in grouped_actuals], rotation=45)
    ax.set_xlabel('Actual Values')
    ax.set_ylabel('Average Errors')

def group_errors_by_actual(errors, actuals):
    error_dict = {}
    for act, err in zip(actuals, errors):
        if act not in error_dict:
            error_dict[act] = []
        error_dict[act].append(err)
    return error_dict

def calculate_average_errors(error_dict):
    avg_pos_errors, avg_neg_errors, grouped_actuals = [], [], sorted(list(set(error_dict.keys())))
    for act in grouped_actuals:
        pos_errors = [e for e in error_dict[act] if e > 0]
        neg_errors = [e for e in error_dict[act] if e < 0]
        avg_pos_errors.append(np.mean(pos_errors) if pos_errors else 0)
        avg_neg_errors.append(np.mean(neg_errors) if neg_errors else 0)
    return avg_pos_errors, avg_neg_errors, grouped_actuals


def main_percentages_():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load datasets
    data1, hurst_params1, coefficients1 = load_from_csv_plus(coef_range=(1.1, 3))
    data2, hurst_params2, coefficients2 = load_from_csv_plus(coef_range=(0.1, 3))
    data2, hurst_params2, coefficients2 = uniform_downsample(len(data1), data2, hurst_params2, coefficients2)

    # Prepare datasets
    tensor_dataset1 = create_tensor_dataset(data1, coefficients1)
    tensor_dataset2 = create_tensor_dataset(data2, coefficients2)

    # Split datasets into training and validation
    train_loader1, val_loader1 = create_data_loaders(tensor_dataset1)
    train_loader2, val_loader2 = create_data_loaders(tensor_dataset2)

    # Initialize models
    model1 = RevisedCNNModel().to(device)
    model2 = RevisedCNNModel().to(device)

    # Train models
    train_model(model1, train_loader1, val_loader1, epochs=44)
    train_model(model2, train_loader2, val_loader2, epochs=44)

    # Evaluate models
    comparative_evaluation(model1, model2, val_loader1, val_loader2)

def main_percentages():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load datasets
    data1_1, hurst_params1_1, coefficients1_1 = load_from_csv_plus(coef_range=(0.1, 2.0))
    data1_2, hurst_params1_2, coefficients1_2 = load_from_csv_plus(coef_range=(0.5, 2.5))
    data1_3, hurst_params1_3, coefficients1_3 = load_from_csv_plus(coef_range=(1.0, 3.0))
    data2, hurst_params2, coefficients2 = load_from_csv_plus(coef_range=(0.1, 3.0))
    data2, hurst_params2, coefficients2 = uniform_downsample(len(data1_1), data2, hurst_params2, coefficients2)

    # Prepare datasets
    tensor_dataset1_1 = create_tensor_dataset(data1_1, coefficients1_1)
    tensor_dataset1_2 = create_tensor_dataset(data1_2, coefficients1_2)
    tensor_dataset1_3 = create_tensor_dataset(data1_3, coefficients1_3)
    tensor_dataset2 = create_tensor_dataset(data2, coefficients2)

    # Split datasets into training and validation
    train_loader1_1, val_loader1_1 = create_data_loaders(tensor_dataset1_1)
    train_loader1_2, val_loader1_2 = create_data_loaders(tensor_dataset1_2)
    train_loader1_3, val_loader1_3 = create_data_loaders(tensor_dataset1_3)
    train_loader2, val_loader2 = create_data_loaders(tensor_dataset2)

    # Initialize models
    model1_1 = RevisedCNNModel().to(device)
    model1_2 = RevisedCNNModel().to(device)
    model1_3 = RevisedCNNModel().to(device)
    model2 = RevisedCNNModel().to(device)

    # Train models
    train_model(model1_1, train_loader1_1, val_loader1_1, epochs=44)
    train_model(model1_2, train_loader1_2, val_loader1_2, epochs=44)
    train_model(model1_3, train_loader1_3, val_loader1_3, epochs=44)
    train_model(model2, train_loader2, val_loader2, epochs=44)

    # Save models
    torch.save(model1_1.state_dict(), 'model1_1_percentages.pth')
    torch.save(model1_2.state_dict(), 'model1_2_percentages.pth')
    torch.save(model1_3.state_dict(), 'model1_3_percentages.pth')
    torch.save(model2.state_dict(), 'model2_percentages.pth')

    # Evaluate models
    comparative_evaluation(model1_1, model2, val_loader1_1, val_loader2)
    comparative_evaluation(model1_2, model2, val_loader1_2, val_loader2)
    comparative_evaluation(model1_3, model2, val_loader1_3, val_loader2)

def main_Hurst():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load datasets
    data1_1, hurst_params1_1, coefficients1_1 = load_from_csv_plus(coef_range=(1.1, 3), hurst_range=(0.1, 0.3))
    data1_2, hurst_params1_2, coefficients1_2 = load_from_csv_plus(coef_range=(1.1, 3), hurst_range=(0.4, 0.6))
    data1_3, hurst_params1_3, coefficients1_3 = load_from_csv_plus(coef_range=(1.1, 3), hurst_range=(0.7, 0.9))
    data2, hurst_params2, coefficients2 = load_from_csv_plus(coef_range=(0.1, 3), hurst_range=(0.0, 0.9))
    data2, hurst_params2, coefficients2 = uniform_downsample(len(data1_1), data2, hurst_params2, coefficients2)

    # Prepare datasets
    tensor_dataset1_1 = create_tensor_dataset(data1_1, coefficients1_1)
    tensor_dataset1_2 = create_tensor_dataset(data1_2, coefficients1_2)
    tensor_dataset1_3 = create_tensor_dataset(data1_3, coefficients1_3)
    tensor_dataset2 = create_tensor_dataset(data2, coefficients2)

    # Split datasets into training and validation
    train_loader1_1, val_loader1_1 = create_data_loaders(tensor_dataset1_1)
    train_loader1_2, val_loader1_2 = create_data_loaders(tensor_dataset1_2)
    train_loader1_3, val_loader1_3 = create_data_loaders(tensor_dataset1_3)
    train_loader2, val_loader2 = create_data_loaders(tensor_dataset2)

    # Initialize models
    model1_1 = RevisedCNNModel().to(device)
    model1_2 = RevisedCNNModel().to(device)
    model1_3 = RevisedCNNModel().to(device)
    model2 = RevisedCNNModel().to(device)

    # Train models
    train_model(model1_1, train_loader1_1, val_loader1_1, epochs=32)
    train_model(model1_2, train_loader1_2, val_loader1_2, epochs=32)
    train_model(model1_3, train_loader1_3, val_loader1_3, epochs=32)
    train_model(model2, train_loader2, val_loader2, epochs=32)

    #save models
    torch.save(model1_1.state_dict(), 'model1_1.pth')
    torch.save(model1_2.state_dict(), 'model1_2.pth')
    torch.save(model1_3.state_dict(), 'model1_3.pth')
    torch.save(model2.state_dict(), 'model2.pth')
    
    # Evaluate models
    comparative_evaluation(model1_1, model2, val_loader1_1, val_loader2)
    comparative_evaluation(model1_2, model2, val_loader1_2, val_loader2)
    comparative_evaluation(model1_3, model2, val_loader1_3, val_loader2)



if __name__ == "__main__":
    main_percentages()

