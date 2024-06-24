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
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.layers import Conv1D, Flatten
from keras.layers import Dense, Dropout, BatchNormalization, LeakyReLU, Reshape
from generator_manager import GeneratorManager
import pandas as pd
import json
import csv
import os
import matplotlib.pyplot as plt
from keras.layers import LSTM
from sklearn.linear_model import LinearRegression
from keras.layers import MaxPooling1D
from sklearn.preprocessing import MinMaxScaler



def main_plotter():
    session, cluster = get_cassandra_session()

    share_ticker = 'AA1'
    initial_share_value = 100
    standard_deviation = 0.1
    standard_deviation = str(standard_deviation)
    # Delete price history for the share ticker
    query = "DELETE FROM shares.price_history WHERE share_ticker = %s"
    session.execute(query, (share_ticker,))

    # Set initial share value
    query = "UPDATE shares.share_details SET share_value = %s WHERE share_ticker = %s"
    session.execute(query, (initial_share_value, share_ticker))

    #change share noise parameter
    query = "UPDATE shares.noise_functions SET noise_parameters = {'standard_deviation': %s} WHERE share_ticker = %s"
    session.execute(query, (standard_deviation, share_ticker))

    # Initialize generator manager and generate data
    generator_manager = GeneratorManager([share_ticker], session)
    generator_manager.generate()

    # Play sound alert
    winsound.PlaySound('Demo/Alert.wav', winsound.SND_FILENAME)

    # Plot price history
    plot_price_history_candlesticks(share_ticker, session)

    # Calculate statistics
    share = Share(share_ticker, session)
    result = share.get_all_values()
    df = pd.DataFrame(list(result))

    df['diff'] = df['value'].diff()

    estimated_sigma = df['diff'].std()
    mad = (df['diff'] - df['diff'].mean()).abs().mean()
    df['rolling_std_50'] = df['diff'].rolling(window=50).std()
    rolling_std_50_avg = df['rolling_std_50'].mean()

    print("Estimated sigma:", estimated_sigma)
    print("MAD:", mad)
    print("Average rolling std_50:", rolling_std_50_avg)

    # Convert Timestamp columns to string for JSON serialization
    df = df.applymap(lambda x: x.isoformat() if isinstance(x, pd.Timestamp) else x)

    # Remove additional columns added for calculation
    df = df.drop(columns=['diff', 'rolling_std_50'])

    # Archive data in JSON
    data_archive = {
        'Data': {
            'AA1': df.to_dict('records')
        }
    }

    with open('share_data.json', 'w') as f:
        json.dump(data_archive, f, indent=4)

    # Archive results in JSON
    results_archive = {
        'Results': {
            'AA1': {
                'Actual Std': standard_deviation,
                'Estimated Std': estimated_sigma,
                'MAD': mad,
                'Average Rolling Std 50': rolling_std_50_avg
            }
        }
    }

    with open('share_results.json', 'w') as f:
        json.dump(results_archive, f, indent=4)

    session.shutdown()


def gauss_singular(session, initial_share_value, standard_deviation, data_file, results_file):
    # Delete price history for the share ticker
    share_ticker = 'AA1'
    query = "DELETE FROM shares.price_history WHERE share_ticker = %s"
    session.execute(query, (share_ticker,))

    # Set initial share value
    query = "UPDATE shares.share_details SET share_value = %s WHERE share_ticker = %s"
    session.execute(query, (initial_share_value, share_ticker))

    # Change share noise parameter
    query = "UPDATE shares.noise_functions SET noise_parameters = {'standard_deviation': %s} WHERE share_ticker = %s"
    session.execute(query, (str(standard_deviation), share_ticker))

    # Initialize generator manager and generate data
    generator_manager = GeneratorManager([share_ticker], session)
    generator_manager.generate()

    # Play sound alert
    #winsound.PlaySound('Demo/Alert.wav', winsound.SND_FILENAME)

    # Plot price history
    #plot_price_history_candlesticks(share_ticker, session)

    # Calculate statistics
    share = Share(share_ticker, session)
    result = share.get_all_values()
    df = pd.DataFrame(list(result))

    df['diff'] = df['value'].diff()

    estimated_sigma = df['diff'].std()
    mad = (df['diff'] - df['diff'].mean()).abs().mean()
    df['rolling_std_50'] = df['diff'].rolling(window=50).std()
    rolling_std_50_avg = df['rolling_std_50'].mean()

    print("Estimated sigma:", estimated_sigma)
    print("MAD:", mad)
    print("Average rolling std_50:", rolling_std_50_avg)

    # Remove additional columns added for calculation
    df = df.drop(columns=['diff', 'rolling_std_50'])

    # Append data to CSV file
    df.to_csv(data_file, mode='a', header=False, index=False)

    # Append results to CSV file
    with open(results_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([initial_share_value, standard_deviation, estimated_sigma, mad, rolling_std_50_avg])

def gauss_main():
    session, cluster = get_cassandra_session()
    
    data_file = 'share_data.csv'
    results_file = 'share_results.csv'

    with open(data_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['timestamp', 'value'])

    with open(results_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['initial_share_value', 'actual_std', 'estimated_std', 'mad', 'average_rolling_std_50'])
    
    for i in range(0, 4):
        initial_share_value = 10**i
        for standard_deviation in np.arange(0.1, 5.1, 0.1):
            gauss_singular(session, initial_share_value, standard_deviation, data_file, results_file)
            gauss_singular(session, initial_share_value, standard_deviation, data_file, results_file)
            gauss_singular(session, initial_share_value, standard_deviation, data_file, results_file)
    
    session.shutdown()


def plot_results_differences(results_file):
    # Load the results CSV into a DataFrame
    results_df = pd.read_csv(results_file)

    # Calculate the differences between actual and estimated values
    results_df['std_diff'] = results_df['estimated_std'] - results_df['actual_std']
    results_df['mad_diff'] = results_df['mad'] - results_df['actual_std']
    results_df['rolling_std_50_diff'] = results_df['average_rolling_std_50'] - results_df['actual_std']

    # Plot the differences
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))

    initial_values = [1, 10, 100, 1000]

    # Plot Estimated Std differences
    for initial_value in initial_values:
        subset = results_df[results_df['initial_share_value'] == initial_value]
        axs[0].plot(subset.index, subset['std_diff'], label=f'Initial Share Value = {initial_value}')
    axs[0].axhline(0, color='black', linewidth=0.5)
    axs[0].set_title('Difference between Actual and Estimated Std')
    axs[0].set_xlabel('Iteration')
    axs[0].set_ylabel('Difference')
    axs[0].legend()
    axs[0].grid(True)

    # Plot MAD differences
    for initial_value in initial_values:
        subset = results_df[results_df['initial_share_value'] == initial_value]
        axs[1].plot(subset.index, subset['mad_diff'], label=f'Initial Share Value = {initial_value}')
    axs[1].axhline(0, color='black', linewidth=0.5)
    axs[1].set_title('Difference between Actual and MAD')
    axs[1].set_xlabel('Iteration')
    axs[1].set_ylabel('Difference')
    axs[1].legend()
    axs[1].grid(True)

    # Plot Average Rolling Std 50 differences
    for initial_value in initial_values:
        subset = results_df[results_df['initial_share_value'] == initial_value]
        axs[2].plot(subset.index, subset['rolling_std_50_diff'], label=f'Initial Share Value = {initial_value}')
    axs[2].axhline(0, color='black', linewidth=0.5)
    axs[2].set_title('Difference between Actual and Average Rolling Std 50')
    axs[2].set_xlabel('Iteration')
    axs[2].set_ylabel('Difference')
    axs[2].legend()
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()


def fBm_export_main():
    session, cluster = get_cassandra_session()
    hurst_values = np.linspace(0.1, 0.9, 9)
    df_all = pd.DataFrame()

    # Generate all share tickers
    share_tickers = [f"{hurst:.1f}".replace('.', '') + f"fBm{i}" for hurst in hurst_values for i in range(1, 101)]

    # List to hold data and corresponding Hurst parameters
    data_list = []
    hurst_params_list = []

    # Loop over each ticker, fetch data, and append to the main DataFrame
    for ticker in share_tickers:
        share = Share(ticker, session)
        result = share.get_all_values()
        df = pd.DataFrame(list(result))
        
        # Extract the Hurst parameter from the ticker name
        hurst_param = float(ticker[:2]) / 10.0  # Convert '09fBm67' to 0.9, etc.

        # Assume 'value' column contains the time series data
        data_series = df['value'].values
        data_list.append(data_series)
        hurst_params_list.append(hurst_param)

    return np.array(data_list), np.array(hurst_params_list)


def extract_features(data):
    features = []
    for series in data:
        mean = np.mean(series)
        std = np.std(series)
        features.append([mean, std, series])

    return np.array(features)



def neural_model_focus1():
    data, hurst_params = fBm_export_main()
    
    # Ensure data is in the correct shape (num_samples, timesteps, features)
    data = np.array(data).astype(float)
    data = data.reshape((data.shape[0], data.shape[1], 1))  # Assuming data is (num_samples, timesteps)
    print(data.shape)
    X_train, X_test, y_train, y_test = train_test_split(data, hurst_params, test_size=0.2, random_state=42)
    
    def create_cnn_model():
        model = Sequential([
            Conv1D(64, kernel_size=3, activation='relu', input_shape=(data.shape[1], 1)),
            MaxPooling1D(pool_size=2),
            Dropout(0.5),
            Conv1D(128, kernel_size=3, activation='relu'),
            MaxPooling1D(pool_size=2),
            Dropout(0.5),
            Flatten(),
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=0.01), loss='mean_squared_error')
        return model

    model = create_cnn_model()

    # Train the model
    history = model.fit(X_train, y_train, epochs=100, validation_split=0.2, batch_size=8, verbose=1)

    # Evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    # Round y_pred to 1 decimal place
    y_pred = np.round(y_pred, 1)
    print(f'Mean Squared Error: {mse}')
    print(f'Predicted Hurst Parameters: {y_pred.flatten()}')
    print(f'Actual Hurst    Parameters: {y_test}')


def neural_model_focus2():
    data, hurst_params = fBm_export_main()
    scaler = MinMaxScaler()
    
    # Normalize data
    data = np.array(data).astype(float)  # Ensure data is float
    data = scaler.fit_transform(data.reshape(-1, 1)).reshape(data.shape)  # Normalize and reshape back
    data = data.reshape((data.shape[0], data.shape[1], 1))  # Reshape for CNN
    print(data.shape)
    
    X_train, X_test, y_train, y_test = train_test_split(data, hurst_params, test_size=0.2, random_state=42)
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")
    def create_cnn_model():
        model = Sequential([
            Conv1D(64, kernel_size=3, activation='relu', input_shape=(data.shape[1], 1)),
            MaxPooling1D(pool_size=2),
            Dropout(0.3),
            Conv1D(128, kernel_size=3, activation='relu'),
            Dropout(0.3),
            Flatten(),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(1, activation='sigmoid') 
        ])
        model.compile(optimizer=Adam(learning_rate=0.0001), loss='mean_squared_error')
        return model

    model = create_cnn_model()
    history = model.fit(X_train, y_train, epochs=100, validation_split=0.2, batch_size=10, verbose=1)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    y_pred = np.round(y_pred, 1)
    print(f'Mean Squared Error: {mse}')
    #print(f'Predicted Hurst Parameters: {y_pred.flatten()}')
    #print(f'Actual Hurst    Parameters: {y_test}')
    #print differences, as a float rounded to 1 decimal place
    print(f'Differences: {np.round(y_test - y_pred.flatten(), 1)}')
    #average of absolute differences
    print(f'Average Absolute Difference: {np.mean(np.abs(y_test - y_pred.flatten()))}')


def linear_regression_model():

    data, hurst_params = fBm_export_main()
    print(data.shape)  # Should be (90, 1000)
    print(hurst_params.shape)  # Should be (90,)

    features = extract_features(data)
    print(features.shape)  # Should be (90, 2)
    print(features[:5])  # Print first 5 to check

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(features, hurst_params, test_size=0.2, random_state=42)
    print(X_train.shape, X_test.shape)  # Check the shapes
    print(y_train.shape, y_test.shape)  # Check the shapes

    

    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate Model
    from sklearn.metrics import mean_squared_error

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')
    print(f'Predicted Hurst Parameters: {y_pred}')
    print(f'Actual Hurst Parameters: {y_test}')


def neural_model_old():
    data, hurst_params = fBm_export_main()

    features = extract_features(data)

    X_train, X_test, y_train, y_test = train_test_split(features, hurst_params, test_size=0.2, random_state=42)


    def create_model():
        model = Sequential([
            Dense(64, activation='relu', input_shape=(2,)),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
        return model
    
    def create_model_deeper():
        model = Sequential([
            Dense(128, activation='relu', input_shape=(2,)),
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
        return model

    def create_model_cnn():
        model = Sequential([
            Reshape((2, 1), input_shape=(2,)),  # Reshaping to (2, 1) for Conv1D layer
            Conv1D(32, 1, activation='relu'),  # Using kernel size 1
            Flatten(),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
        return model
    
    def create_model_leaky():
        model = Sequential([
            Dense(64),
            LeakyReLU(alpha=0.1),
            Dense(32),
            LeakyReLU(alpha=0.1),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
        return model
    
    def create_model_beefy():
        model = Sequential([
            Dense(256, activation='relu', input_shape=(2,)),
            BatchNormalization(),
            Dropout(0.5),
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
        #model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_absolute_error')
        return model

    # Create the model
    #model = create_model()
    #model = create_model_deeper()
    #model = create_model_cnn()
    model = create_model_leaky()

    # Train the model
    history = model.fit(X_train, y_train, epochs=300, validation_split=0.2, batch_size=8, verbose=1)

    # Evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    #round Y_pred to 1 decimal place
    y_pred = np.round(y_pred, 1)
    print(f'Mean Squared Error: {mse}')
    print(f'Predicted Hurst Parameters: {y_pred.flatten()}')
    print(f'Actual Hurst    Parameters: {y_test}')


def neural_model():
    data, hurst_params = fBm_export_main()
    features = extract_features(data)

    X_train, X_test, y_train, y_test = train_test_split(features, hurst_params, test_size=0.2, random_state=42)

    def create_model():
        model = Sequential([
            Dense(64, activation='relu', input_shape=(2,)),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
        return model
    
    def create_model_deeper():
        model = Sequential([
            Dense(128, activation='relu', input_shape=(2,)),
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
        return model

    def create_model_cnn():
        model = Sequential([
            Reshape((2, 1), input_shape=(2,)),
            Conv1D(32, 1, activation='relu'),
            Flatten(),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
        return model
    
    def create_model_leaky():
        model = Sequential([
            Dense(64),
            LeakyReLU(alpha=0.1),
            Dense(32),
            LeakyReLU(alpha=0.1),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
        return model
    
    def create_model_beefy():
        model = Sequential([
            Dense(256, activation='relu', input_shape=(2,)),
            BatchNormalization(),
            Dropout(0.5),
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_absolute_error')
        return model

    models = {
        "simple": create_model,
        "deeper": create_model_deeper,
        "cnn": create_model_cnn,
        "leaky": create_model_leaky,
        "beefy": create_model_beefy
    }

    os.makedirs("early_nn", exist_ok=True)

    results = {}
    
    for name, create_fn in models.items():
        model = create_fn()
        history = model.fit(X_train, y_train, epochs=300, validation_split=0.2, batch_size=8, verbose=0)
        
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        
        results[name] = {
            "mse": mse,
            "predictions": y_pred.flatten().tolist(),
            "actual": y_test.tolist()
        }

        model.save(f"early_nn/model_{name}.h5")

    with open("early_nn/predictions_and_performance.json", "w") as f:
        json.dump(results, f, indent=4)

    df_all = pd.DataFrame(data)
    df_all.to_csv("early_nn/fBm_data.csv", index=False)


def resave_data_with_hurst():
    data, hurst_params = fBm_export_main()

    # Create a DataFrame from the data and Hurst parameters
    df_data = pd.DataFrame(data)
    df_data['hurst_params'] = hurst_params

    # Save the DataFrame to a CSV file
    os.makedirs("early_nn", exist_ok=True)
    df_data.to_csv("early_nn/fBm_data_with_hurst.csv", index=False)


def neural_model_focus():
    data, hurst_params = fBm_export_main()
    scaler = MinMaxScaler()

    # Normalize data
    data = np.array(data).astype(float)  # Ensure data is float
    data = scaler.fit_transform(data.reshape(-1, 1)).reshape(data.shape)  # Normalize and reshape back
    data = data.reshape((data.shape[0], data.shape[1], 1))  # Reshape for CNN
    print(data.shape)
    
    X_train, X_test, y_train, y_test = train_test_split(data, hurst_params, test_size=0.2, random_state=42)
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")

    def create_cnn_model():
        model = Sequential([
            Conv1D(64, kernel_size=3, activation='relu', input_shape=(data.shape[1], 1)),
            MaxPooling1D(pool_size=2),
            Dropout(0.3),
            Conv1D(128, kernel_size=3, activation='relu'),
            Dropout(0.3),
            Flatten(),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(1, activation='sigmoid') 
        ])
        model.compile(optimizer=Adam(learning_rate=0.0001), loss='mean_squared_error')
        return model

    model = create_cnn_model()
    history = model.fit(X_train, y_train, epochs=100, validation_split=0.2, batch_size=16, verbose=1)

    # Plotting the training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss Progress During Training')
    plt.ylabel('Mean Squared Error')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    y_pred = np.round(y_pred, 1)
    print(f'Mean Squared Error: {mse}')
    print(f'Differences: {np.round(y_test - y_pred.flatten(), 1)}')
    print(f'Average Absolute Difference: {np.mean(np.abs(y_test - y_pred.flatten()))}')


if __name__ == "__main__":
    #print(extract_features(fBm_export_main()))
    #linear_regression_model()
    #neural_model()
    #resave_data_with_hurst()
    gauss_main()
    #plot_results_differences('./_Results_Archive/gauss_results.csv')
    #neural_model_old()
    #neural_model_focus()
