import datetime
import time
from share import Share
import numpy as np
from database import get_cassandra_session

"""
TABLE shares.linkage_functions (
        primary_share_ticker TEXT,
        secondary_share_ticker TEXT,
        linkage_id UUID,
        linkage_function_type TEXT,
        linkage_parameters MAP<TEXT, TEXT>,
        PRIMARY KEY (linkage_id)
);
"""

# Represents a linkage function that models the interaction between two stocks.
class LinkageFunction:
    # Constructor to initialize a LinkageFunction with its database identifier and session.
    def __init__(self, linkage_id, session):
        self.linkage_id = linkage_id
        self.session = session
        self.secondary_share_values = []
        self.secondary_share_values_slice = []
        self._load_function()
        self.primary_share_value = None

    # Returns a list of all possible types of linkage functions.
    @staticmethod
    def get_linkage_function_types():
        return ['linear', 'exponential', 'quadratic', 'logarithmic', 'momentum', 'diminishing_returns_momentum', 'competitor_momentum', 'percentage_change', 'diminishing_return_percentage_change', 'competitor_percentage_change']

    # Returns the required parameters for a specified type of linkage function.
    @staticmethod
    def get_parameter_spec(linkage_function_type):
        specs = {
            'linear': ['coefficient'],
            'exponential': ['coefficient', 'base'],
            'quadratic': ['coefficient', 'exponent'],
            'logarithmic': ['coefficient', 'base'],
            'momentum': ['lookback_period', 'coefficient'],
            'diminishing_returns_momentum': ['lookback_period', 'coefficient'],
            #'percentage_change': ['lookback_period', 'coefficient'],
            'percentage_change': ['coefficient'],
            'diminishing_return_percentage_change': ['lookback_period', 'coefficient'],
            'competitor_percentage_change': ['lookback_period', 'coefficient']
        }
        return specs.get(linkage_function_type, [])  # Fetch and return the parameter specification for the function type.

    # Loads the linkage function details from the database and initializes the object's attributes.
    def _load_function(self):
        query = "SELECT primary_share_ticker, secondary_share_ticker, linkage_function_type, linkage_parameters FROM shares.linkage_functions WHERE linkage_id = %s"
        result = self.session.execute(query, (self.linkage_id,))
        row = result.one()  # Fetches one result from the query.
        if row is not None:
            self.linkage_function_type = row.linkage_function_type
            self.primary_share_ticker = row.primary_share_ticker
            self.secondary_share_ticker = row.secondary_share_ticker
            query = "SELECT timestamp, value FROM shares.price_history WHERE share_ticker = %s"
            result = self.session.execute(query, (self.secondary_share_ticker,))
            for row2 in result:
                self.secondary_share_values.append((row2.timestamp, row2.value))
            self.secondary_share_values.sort(key=lambda x: x[0], reverse=False)
            print (f"Loaded {len(self.secondary_share_values)} values for {self.secondary_share_ticker}")
            self.linkage_parameters = row.linkage_parameters
            for key in self.linkage_parameters:
                setattr(self, key, float(self.linkage_parameters[key]))  # Convert string parameters to floats and set them as attributes.
        else:
            # Handle the case where no data is found for the linkage function.
            self.linkage_function_type = None
            self.primary_share_ticker = None
            self.secondary_share_ticker = None
            self.linkage_parameters = None



    # Initializes the parameters for the linkage function based on the specified types.
    def _init_linkage_function_parameters(self):
        params = self.get_parameter_spec(self.linkage_function_type)
        for param in params:
            setattr(self, param, self.linkage_parameters.get(param, None))  # Set each parameter using the linkage_parameters map.

    # Calculates the effect of the linkage on the primary share value at a given timestamp.
    def calculate_effect(self, timestamp, primary_share_value):
        self.primary_share_value = primary_share_value
        # if self.lookback_period == None:
        #     self.secondary_share_values_slice = self.get_secondary_share_values(timestamp, 2)
        # else:
        #     self.secondary_share_values_slice = self.get_secondary_share_values(timestamp, max(self.lookback_period, 2))
        self.secondary_share_values_slice = self.get_secondary_share_values(timestamp, 2)
        # try:
        #     # Calculate rate of change based on historical values.
        #     #rate_of_change = (self.secondary_share_values_slice[0] - self.secondary_share_values_slice[-1]) / len(secondary_share_values)
        #     #print (f"Rate of change for {self.secondary_share_ticker}: {rate_of_change}")
        # except:
        #     print (f"Error calculating rate of change for {self.secondary_share_ticker}")
        #     return 0  # Return 0 on error to prevent crashing.
        print(self.secondary_share_values_slice)
        # Determine which type of linkage function to apply based on the defined type.
        if self.linkage_function_type == 'linear':
            return self._linear_linkage()
        elif self.linkage_function_type == 'exponential':
            return self._exponential_linkage()
        elif self.linkage_function_type == 'quadratic': 
            return self._quadratic_linkage()
        elif self.linkage_function_type == 'logarithmic':
            return self._logarithmic_linkage()
        elif self.linkage_function_type == 'momentum':
            return self._momentum_linkage()
        elif self.linkage_function_type == 'diminishing_return_momentum':
            return self._diminishing_returns_momentum()
        elif self.linkage_function_type == 'percentage_change':
            return self._percentage_change_linkage()
        elif self.linkage_function_type == 'diminishing_return_percentage_change':
            return self._diminishing_returns_percentage_change()
        else:
            print(f'Invalid linkage function type {self.linkage_function_type}')
            return 0  # Return zero for an undefined linkage function type as a fail-safe.

    # Applies a linear relationship to determine the effect on the primary share value.
    def _linear_linkage(self):
        return self.coefficient * self.secondary_share_values_slice[0]

    # Applies an exponential relationship based on the base and coefficient parameters.
    def _exponential_linkage(self):
        return self.coefficient * (self.base ** self.secondary_share_values_slice)

    # Applies a quadratic relationship, utilizing the coefficient and exponent parameters.
    def _quadratic_linkage(self):
        return self.coefficient * (self.secondary_share_values_slice ** self.exponent)

    # Applies a logarithmic relationship, taking into account the coefficient and base.
    def _logarithmic_linkage(self):
        return self.coefficient * np.log(self.secondary_share_values_slice) / np.log(self.base) 

    # Calculates the impact based on the momentum of change in the secondary share values.
    def _momentum_linkage(self):
        rate_of_change = (self.secondary_share_values_slice[0] - self.secondary_share_values_slice[-1]) / len(self.secondary_share_values_slice)
        return self.coefficient * rate_of_change
    
    # Modifies the basic momentum effect by applying a logarithmic function to simulate diminishing returns.
    def _diminishing_returns_momentum(self):
        rate_of_change = (self.secondary_share_values_slice[0] - self.secondary_share_values_slice[-1]) / len(self.secondary_share_values_slice)
        effect = self.coefficient * rate_of_change
        # Uses a logarithmic function to modify the effect, thus simulating diminishing returns.
        log = (abs(effect) / (1 + abs(effect)))
        return effect * log

    # Calculates the effect based on the percentage rate of change in the secondary share values.
    def _percentage_change_linkage(self):
        percentage_change = ((self.secondary_share_values_slice[-1] - self.secondary_share_values_slice[-2]) / self.secondary_share_values_slice[-2]) * 100
        return (self.primary_share_value * (self.coefficient * percentage_change)) / 100

    # Similar to momentum but calculates the effect based on a percentage change and applies diminishing returns.
    def _diminishing_returns_percentage_change(self):
        rate_of_change = (self.secondary_share_values_slice[0] - self.secondary_share_values_slice[-1]) / self.secondary_share_values_slice[-1]
        effect = self.coefficient * rate_of_change
        # Modifies the effect using a logarithmic function to simulate diminishing returns.
        log = (abs(effect) / (1 + abs(effect)))
        return effect * log
    

    def get_secondary_share_values(self, timestamp, lookback_period = 2):
        # Convert the passed timestamp from seconds since Epoch to datetime.datetime
        timestamp_datetime = datetime.datetime.fromtimestamp(timestamp)
        # Filter to get values where timestamps are <= the passed datetime
        filtered_values = [value for ts, value in self.secondary_share_values if ts <= timestamp_datetime]
        # Return the last 'lookback_period' number of values from the filtered list
        if len(filtered_values) < 2:
            return [1, 1]
        return filtered_values[int(-lookback_period):]




