import threading
from database import get_cassandra_session
import time
from generator import Generator
from gui import plot_price_history_candlesticks, plot_two_price_histories, plot_multiple_price_histories
import winsound
import pandas as pd
from share import Share



class GeneratorManager:
    def __init__(self, share_tickers, session):
        self.generators = {ticker: Generator(ticker, session) for ticker in share_tickers}
        self.session = session
        self.count = 0

    def generate(self):
        timestamp_start = 1546300800 # 2019-01-01
        #timestamp_end = 1546300800 + 86400 * 365 * 2  # 2020-12-31
        timestamp_end = 1672464000   # 2022-12-31
        # Process one ticker at a time across all timestamps

        for ticker, generator in self.generators.items():
            #print(f"Starting generation for {ticker}")
            self.generate_for_ticker(generator, timestamp_start, timestamp_end)
            generator.share.batch_insert_price_history()
            self.count += 1
            print(self.count)

    def generate_for_ticker(self, generator, start, end):
        timestamp = start
        while timestamp < end:
            if generator.share.share_value == -1:
                timestamp += 86400
                continue
            generator.timestamp = timestamp
            generator._generate_and_callback_data()
            timestamp += 86400
            #print(time.strftime('%Y-%m-%d', time.localtime(timestamp)))
    

def gauss_main():
    session, cluster = get_cassandra_session()
    share1 = Share('AA1', session)
    share2 = Share('AA2', session)
    result1 = share1.get_all_values()
    result2 = share2.get_all_values()
    df1 = pd.DataFrame(list(result1))
    df2 = pd.DataFrame(list(result2))

    df1['diff'] = df1['value'].diff()
    df2['diff'] = df2['value'].diff()


    estimated_sigma1 = df1['diff'].std()
    estimated_sigma2 = df2['diff'].std()

    df1['rolling_std_50'] = df1['diff'].rolling(window=50).std()
    df2['rolling_std_50'] = df2['diff'].rolling(window=50).std()

    # Median Absolute Deviation (MAD)
    #df1_mad = df1['diff'].mad()
    df1_mad = (df1['diff'] - df1['diff'].mean()).abs().mean()
    #df2_mad = df2['diff'].mad()
    df2_mad = (df2['diff'] - df2['diff'].mean()).abs().mean()


    # Calculate the average of the rolling std windows
    rolling_std_50_avg_1 = df1['rolling_std_50'].mean()
    rolling_std_50_avg_2 = df2['rolling_std_50'].mean()

    print("Estimated sigma1:", estimated_sigma1)
    print("Estimated sigma2:", estimated_sigma2)
    print("df1 MAD:", df1_mad)
    print("df2 MAD:", df2_mad)
    print("Average rolling std_50 for df1:", rolling_std_50_avg_1)
    print("Average rolling std_50 for df2:", rolling_std_50_avg_2)
    session.shutdown()


def testing():
    session, cluster = get_cassandra_session()

    share_tickers = ['00Hurst']
    for ticker in share_tickers:
        query = "DELETE FROM shares.price_history WHERE share_ticker = %s"
        session.execute(query, (ticker,))
    initial_share_values = [100]
    for ticker, value in zip(share_tickers, initial_share_values):
        query = "UPDATE shares.share_details SET share_value = %s WHERE share_ticker = %s"
        session.execute(query, (value, ticker))

    print ("Starting generator manager iinnn 3 2 1...")
    generator_manager = GeneratorManager(share_tickers, session)
    generator_manager.generate()

    plot_two_price_histories('XLK', '00Hurst', session)
    session.shutdown()

if __name__ == "__main__":
    testing()