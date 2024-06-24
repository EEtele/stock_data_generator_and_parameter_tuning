import csv
import os
import matplotlib.pyplot as plt
from datetime import datetime


def create_gics_folders():
    # List of GICS sectors
    gics_sectors = [
        'Energy',
        'Materials',
        'Industrials',
        'Consumer Discretionary',
        'Consumer Staples',
        'Health Care',
        'Financials',
        'Information Technology',
        'Communication Services',
        'Utilities',
        'Real Estate'
    ]

    # Directory where the folders will be created
    base_directory = ''

    # Create the folders
    for sector in gics_sectors:
        folder_name = sector.replace(' ', '_')  # Replace spaces with underscores
        folder_path = os.path.join(base_directory, folder_name)
        os.makedirs(folder_path)

    print('Folders created successfully.')




def plot_csv_data(csv_file, y_column):
    timestamps = []
    values = []

    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row

        for row in reader:
            timestamp = datetime.strptime(row[0], "%Y-%m-%d")
            value = float(row[y_column])
            timestamps.append(timestamp)
            values.append(value)

    plt.plot(timestamps, values)
    plt.xlabel('Timestamp')
    plt.ylabel(f'Value at column {y_column}')
    plt.title('CSV Data Plot')
    plt.xticks(rotation=45)
    plt.show()


def plot_all_files(folder_path, y_column):
    for file in os.listdir(folder_path):
        if file.endswith(".csv"):
            plot_csv_data(os.path.join(folder_path, file), y_column)


import os

def plot_csv_files(folder, y_column=1):
    filenames = []
    data = []

    # Walk through the folder and find CSV files
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith(".csv"):
                filenames.append(file)
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as file:
                    reader = csv.reader(file)
                    headers = reader.__next__()  # Get the header row
                    #next(reader)  # Skip the header row

                    timestamps = []
                    values = []

                    for row in reader:
                        timestamp = datetime.strptime(row[0], "%Y-%m-%d")
                        value = float(row[y_column])
                        timestamps.append(timestamp)
                        values.append(value)

                    data.append((timestamps, values))

    # Plotting
    plt.figure(figsize=(12, 8))

    for i, (timestamps, values) in enumerate(data):
        label = filenames[i]
        plt.plot(timestamps, values, label=label)

    plt.xlabel('Timestamp')
    #column header
    plt.ylabel(f'Value at column {headers[y_column]}')
    plt.title(f'{folder} CSV Data Plot')
    plt.xticks(rotation=45)
    plt.legend()
    plt.show()




def modify_csv_files(folder):
    # Walk through the folder and find CSV files
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith(".csv"):
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as file:
                    reader = csv.reader(file)
                    header = next(reader)  # Get the header row

                    date_index = header.index("Date")
                    adj_close_index = header.index("Adj Close")

                    rows = []
                    rows.append(["Date", "Adj Close"])  # Add the modified header row

                    for row in reader:
                        modified_row = [row[date_index], row[adj_close_index]]
                        rows.append(modified_row)

                # Write the modified data back to the CSV file
                with open(file_path, 'w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerows(rows)





if __name__ == "__main__": 
    #plot by directory
    for root, dirs, files in os.walk("Stocks"):
        for directory in dirs:
            plot_csv_files(os.path.join(root, directory))

    


