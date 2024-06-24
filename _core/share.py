from database import get_cassandra_session
import pandas as pd
import time
from cassandra.query import BatchStatement, SimpleStatement
from cassandra.concurrent import execute_concurrent_with_args
from datetime import datetime, timedelta
"""
shares.share_details (
        share_ticker TEXT,
        share_name TEXT,
        share_description TEXT,
        share_value DOUBLE,
        share_events MAP<TEXT, TEXT>,
        PRIMARY KEY (share_ticker)
);
"""

class Share:
    def __init__(self, share_ticker, session, share_name=None, share_value=None, share_events=None, create=False):
        self.session = session
        self.share_ticker = share_ticker
        self.share_name = share_name
        self.share_value = share_value
        self.share_events = share_events
        self.array = []
        self.prepared_query = self.session.prepare("INSERT INTO shares.price_history (share_ticker, timestamp, value) VALUES (?, ?, ?)")

        if create:
            self.create_share() 
        elif share_name is None or share_value is None or share_events is None:
            self._load_share_data()
        else:
            print(f"Initialized share {self.share_ticker} without loading or creating.")

    def _load_share_data(self):
        query = "SELECT * FROM shares.share_details WHERE share_ticker = %s"
        result = self.session.execute(query, (self.share_ticker,))
        if result.one() is None:
            raise ValueError(f"No share found with ticker '{self.share_ticker}'")
        else:
            share_data = result.one()
            self.share_name = share_data.share_name
            self.share_value = share_data.share_value
            self.share_events = share_data.share_events
            print(f"Share {self.share_ticker} loaded")

    def create_share(self):
        # Check if the share already exists
        query = "SELECT share_ticker FROM shares.share_details WHERE share_ticker = %s"
        result = self.session.execute(query, (self.share_ticker,))
        if result.one() is not None:
            raise ValueError(f"Share with ticker '{self.share_ticker}' already exists.")
        
        # Insert new share if it does not exist
        query = "INSERT INTO shares.share_details (share_ticker, share_name, share_value, share_events) VALUES (%s, %s, %s, %s)"
        self.session.execute(query, (self.share_ticker, self.share_name, self.share_value, self.share_events))
        print(f"Share {self.share_ticker} created")

    def add_noise_function(self, noise_function_type, noise_parameters):
        query = "INSERT INTO shares.noise_functions (share_ticker, noise_function_type, noise_parameters) VALUES (%s, %s, %s)"
        self.session.execute(query, (self.share_ticker, noise_function_type, noise_parameters))

    def refresh(self):
        self._load_share_data()

    def __str__(self):
        return (f"Share Ticker: {self.share_ticker}\n"
                f"Share Name: {self.share_name}\n"
                f"Share Value: {self.share_value}\n"
                f"Share Events: {self.share_events}")

    def _update_share_field(self, field_name, field_value):
        query = f"UPDATE shares.share_details SET {field_name} = %s WHERE share_ticker = %s"
        self.session.execute(query, (field_value, self.share_ticker))

    def set_share_name(self, share_name):
        self._update_share_field('share_name', share_name)
        self.share_name = share_name

    def set_share_value(self, share_value, timestamp):
        self.share_value = share_value
        self.array.append((timestamp, share_value))
        

    def set_share_events(self, share_events):
        self._update_share_field('share_events', share_events)
        self.share_events = share_events



    def batch_insert_price_history(self):
        if not self.array:
            print("No new values to insert.")
            return
        
        batch = BatchStatement()
        query = SimpleStatement("INSERT INTO shares.price_history (share_ticker, timestamp, value) VALUES (%s, %s, %s)")
        
        for timestamp, value in self.array:
            batch.add(query, (self.share_ticker, timestamp, value))
        
        self.session.execute(batch)
        self.array.clear()
        #print(f"Inserted {len(batch)} records into shares.price_history for {self.share_ticker}")

    def to_dict(self):
        return {
            'share_ticker': self.share_ticker,
            'share_name': self.share_name,
            'share_value': self.share_value,
            'share_events': self.share_events
        }
    
    def get_share_value_at_time(self, timestamp):
        timestamp_datetime = time.strftime('%Y-%m-%d', time.localtime(timestamp))
        query = "SELECT value FROM shares.price_history WHERE share_ticker = %s AND timestamp < %s ORDER BY timestamp DESC LIMIT 1"
        #print(query, (self.share_ticker, timestamp_datetime))
        result = self.session.execute(query, (self.share_ticker, timestamp_datetime))
        row = result.one()
        return row.value if row else None
    
    def get_last_x_values(self, timestamp, x):
        timestamp_datetime = time.strftime('%Y-%m-%d', time.localtime(timestamp))
        x=int(x)
        query = "SELECT value FROM shares.price_history WHERE share_ticker = %s AND timestamp < %s ORDER BY timestamp DESC LIMIT %s"
        result = self.session.execute(query, (self.share_ticker, timestamp_datetime, x))
        return result if result else None
    
    def get_all_values(self):
        query = "SELECT timestamp, value FROM shares.price_history WHERE share_ticker = %s"
        result = self.session.execute(query, (self.share_ticker,))
        return result if result else None
        # df = pd.DataFrame(list(result))
        # return df

    def fill_missing_dates_automatically(self):
        # First, determine the date range from the existing data
        query = "SELECT MIN(timestamp), MAX(timestamp) FROM shares.price_history WHERE share_ticker = %s"
        result = self.session.execute(query, (self.share_ticker,))
        row = result.one()
        if row is None or row[0] is None or row[1] is None:
            print("No existing data to determine date range.")
            return

        # No need to parse as datetime objects are already returned
        min_date = row[0]
        max_date = row[1]

        # Retrieve all existing values
        query = "SELECT timestamp, value FROM shares.price_history WHERE share_ticker = %s ORDER BY timestamp"
        result = self.session.execute(query, (self.share_ticker,))
        values_dict = {row.timestamp: row.value for row in result}

        # Initialize variables for filling
        last_value = None
        filled_values = []

        # Fill missing dates
        for single_date in self.date_range(min_date, max_date):
            if single_date in values_dict:
                last_value = values_dict[single_date]
            elif last_value is not None:
                filled_values.append((single_date.strftime('%Y-%m-%d'), last_value))

        # Optionally insert filled values back into the database
        for date_str, value in filled_values:
            self.set_share_value(value, datetime.strptime(date_str, '%Y-%m-%d'))
        self.batch_insert_price_history()
        print(f"Filled and inserted {len(filled_values)} missing records.")

    def fill_and_pad_missing_dates(self):
        # Determine the date range from the existing data
        query = "SELECT MIN(timestamp), MAX(timestamp) FROM shares.price_history WHERE share_ticker = %s"
        result = self.session.execute(query, (self.share_ticker,))
        row = result.one()
        if row is None or row[0] is None or row[1] is None:
            print("No existing data to determine date range.")
            return

        # Extend the range to include the day before and after
        min_date = row[0] - timedelta(days=1)
        max_date = row[1] + timedelta(days=1)

        # Retrieve all existing values
        query = "SELECT timestamp, value FROM shares.price_history WHERE share_ticker = %s ORDER BY timestamp"
        result = self.session.execute(query, (self.share_ticker,))
        values_dict = {row.timestamp: row.value for row in result}

        # Determine first and last values explicitly
        if values_dict:
            first_value = next(iter(values_dict.values()))
            last_value = list(values_dict.values())[-1]
        else:
            first_value = last_value = None

        # Initialize for filling and padding
        filled_values = []
        last_known_value = None

        # Fill and pad missing dates
        for single_date in self.date_range(min_date, max_date):
            if single_date in values_dict:
                last_known_value = values_dict[single_date]
            elif last_known_value is not None:
                filled_values.append((single_date.strftime('%Y-%m-%d'), last_known_value))

        # Add padding for the first and last day if they were not covered in the loop
        if min_date not in values_dict and first_value:
            filled_values.append((min_date.strftime('%Y-%m-%d'), first_value))
        if max_date not in values_dict and last_value:
            filled_values.append((max_date.strftime('%Y-%m-%d'), last_value))

        # Optionally insert filled and padded values back into the database
        for date_str, value in filled_values:
            self.set_share_value(value, datetime.strptime(date_str, '%Y-%m-%d'))
        self.batch_insert_price_history()
        print(f"Filled and inserted {len(filled_values)} missing records.")

    @staticmethod
    def date_range(start_date, end_date):
        for n in range(int((end_date - start_date).days) + 1):
            yield start_date + timedelta(n)

def get_all_shares_simple(session):
    query = "SELECT share_ticker FROM shares.share_details"
    results = session.execute(query)
    shares = [Share(row.share_ticker, session) for row in results]
    return shares

def get_all_shares_real(session):
    # Modified query to select only shares where share_value is -1
    query = "SELECT share_ticker FROM shares.share_details WHERE share_value = -1 ALLOW FILTERING"
    results = session.execute(query)
    shares = [Share(row.share_ticker, session) for row in results]
    return shares

# gets all 'real' shares and fills and pads missing dates
def fill_all_shares(session):
    shares = get_all_shares_real(session)
    for share in shares:
        share.fill_and_pad_missing_dates()

