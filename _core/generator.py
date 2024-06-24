from noise import NoiseFunction
from linkage import LinkageFunction
from share import Share
import time

class Generator:
    # Initializes the Generator with a share ticker, a database session, and an optional callback function.
    def __init__(self, share_ticker, session, data_callback=None):
        self.timestamp = 1546300800  # Start timestamp for generating data (January 1, 2019).
        self.session = session
        self.share = Share(share_ticker, session)
        self.noise_function = NoiseFunction(share_ticker, session)  # Initializes the NoiseFunction object.
        self.data_callback = data_callback if data_callback else self.default_data_callback  # If no callback is provided, use the default data callback method.
        self._load_linkage_functions(self.session)  # Load linkage functions from the database.


    
    def default_data_callback(self, new_timestamp, new_share_value):
        print(f"Share Ticker: {self.share.share_ticker}, Value: {round(new_share_value, 3)}, Timestamp: {new_timestamp}")
        self.share.set_share_value(new_share_value, new_timestamp)

    def slow_data_callback(self):
        print(f"Share Ticker: {self.share.share_ticker}, Value: {round(self.share.share_value, 3)}")
        timestamp_datetime = time.strftime('%Y-%m-%d', time.localtime(self.timestamp))  # Convert timestamp to date string.
        query = "INSERT INTO shares.price_history (share_ticker, timestamp, value) VALUES (%s, %s, %s)"
        self.session.execute(query, (self.share.share_ticker, timestamp_datetime, self.share.share_value))

    # Loads linkage functions from the database that affect the stock's price.
    def _load_linkage_functions(self, session):
        query = "SELECT linkage_id FROM shares.linkage_functions WHERE primary_share_ticker = %s ALLOW FILTERING"
        results = session.execute(query, (self.share.share_ticker,))
        self.linkage_functions = [LinkageFunction(row.linkage_id, session) for row in results]

    # Generates new stock price data until a specified end timestamp (December 31, 2022).
    def generate(self):
        while self.timestamp < 1672464000:
            self._generate_and_callback_data()  # Generate and handle new data.
            self.timestamp += 86400  # Increment the day (86400 seconds = 1 day).
            print(time.strftime('%Y-%m-%d', time.localtime(self.timestamp)))  # Print the new date.

    # Handles the generation and updating of the stock price.
    def _generate_and_callback_data(self):
        new_share_value = self.share.share_value
        timestamp_datetime = time.strftime('%Y-%m-%d', time.localtime(self.timestamp))
        if new_share_value == -1:
            value = self.share.get_share_value_at_time(self.timestamp)
            print(f"Share Ticker: {self.share.share_ticker}, Value: {round(value, 3)}")
            return

        for linkage_function in self.linkage_functions:
            new_share_value += linkage_function.calculate_effect(self.timestamp, self.share.share_value)

        new_share_value = self.noise_function.calculate_noisy_value(new_share_value)
        new_share_value = max(new_share_value, 1)  # Ensure the share value does not drop below 1
        
        self.data_callback(timestamp_datetime, new_share_value)
