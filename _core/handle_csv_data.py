import os
import pandas as pd
from database import get_cassandra_session
import datetime
import uuid



def process_file(file_path, session):
    share_ticker = os.path.basename(file_path).replace(".csv", "")
    df = pd.read_csv(file_path)

    query = session.prepare("INSERT INTO shares.price_history (share_ticker, timestamp, value) VALUES (?, ?, ?);")

    futures = []

    for index, row in df.iterrows():
        timestamp = datetime.datetime.strptime(row['Date'], '%Y-%m-%d')
        value = float(row['Adj Close'])

        future = session.execute_async(query, (share_ticker, timestamp, value))
        futures.append(future)

    for future in futures:
        try:
            future.result()
        except Exception as e:
            print(f"An error occurred: {e}")

    return share_ticker


def process_all_files(dir_path = "Stocks", session = None):
    if session is None:
        session, cluster = get_cassandra_session()
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if file.endswith(".csv"):
                file_path = os.path.join(root, file)
                share_ticker = process_file(file_path, session= session)
                print(f"Processed {file_path}")

                sector = os.path.basename(os.path.dirname(file_path))

                query = session.prepare("INSERT INTO shares.share_details (share_ticker, share_name, share_description, share_value, share_events) VALUES (?, ?, ?, ?, ?);")
                session.execute(query, (share_ticker, share_ticker, sector, -1, {}))
                print(f"Inserted share details for {share_ticker}")

    print("Done")

def main():
    session, cluster = get_cassandra_session()

    dir_path = "Stocks"

    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if file.endswith(".csv"):
                file_path = os.path.join(root, file)
                share_ticker = process_file(file_path, session= session)
                print(f"Processed {file_path}")
                sector = os.path.basename(os.path.dirname(file_path))
                query = session.prepare("INSERT INTO shares.share_details (share_ticker, share_name, share_description, share_value, share_events) VALUES (?, ?, ?, ?, ?);")
                session.execute(query, (share_ticker, share_ticker, sector, -1, {}))
                print(f"Inserted share details for {share_ticker}")

    print("Done")

    session.shutdown()
    cluster.shutdown()


if __name__ == "__main__":
    main()