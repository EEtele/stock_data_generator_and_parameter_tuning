import random
import numpy as np
from database import get_cassandra_session

# Defines the database schema for storing noise function configurations for stocks.
"""
TABLE shares.noise_functions (
        share_ticker TEXT,
        noise_function_type TEXT,
        noise_parameters MAP<TEXT, TEXT>,
        PRIMARY KEY (share_ticker)
);
"""

# Class to model different types of noise in stock prices based on predefined functions.
class NoiseFunction:
    # Initializes a NoiseFunction object for a specific stock ticker.
    def __init__(self, share_ticker, session):
        self.share_ticker = share_ticker  # Stock ticker for which to model noise.
        self.session = session  # Database session for querying noise function data.
        self._load_function()  # Load the noise function configuration from the database.
        self.number_of_increments = 1  # Default number of increments, modified later if needed.
        self._init_noise_function_parameters()  # Initialize parameters specific to the loaded noise function.

    # Loads the noise function type and parameters from the database.
    def _load_function(self):
        query = "SELECT noise_function_type, noise_parameters FROM shares.noise_functions WHERE share_ticker = %s"
        result = self.session.execute(query, (self.share_ticker,))
        row = result.one()  # Retrieves the noise function configuration if it exists.
        if row is not None:
            self.noise_function_type = row.noise_function_type
            self.noise_parameters = row.noise_parameters    
        else:
            # Set defaults if no configuration is found.
            self.noise_function_type = None
            self.noise_parameters = None

    # Returns a list of possible noise function types that can be applied to stock prices.
    @staticmethod
    def get_noise_function_types():
        return ['uniform_percentage', 'uniform', 'gaussian', 'gaussian_percentage', 'heston_percentage', 'heston_absolute', 'FBM']

    # Returns the required parameters for a given noise function type.
    @staticmethod
    def get_parameter_spec(function_type):
        specs = {
            'uniform_percentage': ['percentage_deviation'],
            'uniform': ['deviation'],
            'gaussian': ['standard_deviation'],
            'gaussian_percentage': ['standard_deviation'],
            'FBM': ['hurst_parameter']
        }
        return specs.get(function_type, [])  # Fetches parameter specs based on the function type.

    # Initializes noise function parameters from the stored configuration.
    def _init_noise_function_parameters(self):
        params = self.get_parameter_spec(self.noise_function_type)
        for param in params:
            value = self.noise_parameters.get(param, None)  # Retrieve parameter value if it exists.
            if value is not None:
                value = float(value)  # Convert parameter value to float for numerical calculations.
            setattr(self, param, value)  # Set the attribute on the object dynamically.

        # Special handling for FBM (Fractional Brownian Motion) noise type.
        if self.noise_function_type == 'FBM':
            self.length = 1500  # Length of the sequence for simulation purposes.
            self.number_of_increments = 0  # Adjust number of increments for this type.
            self.noise_sequence = self._FBM_sequence()  # Generate the FBM sequence.





        # Applies uniform percentage-based noise to a given value based on a specified deviation percentage.
    def _uniform_percentage_based_noise(self, value):
        deviation = value * self.percentage_deviation / 100  # Calculate the deviation as a percentage of the value.
        return value + random.uniform(-deviation, deviation)  # Add a random deviation within the calculated range to the value.

    # Applies uniform noise with a specified deviation from the original value.
    def _uniform_noise(self, value):
        return value + random.uniform(-self.deviation, self.deviation)  # Add a random deviation, defined by the 'deviation' attribute.

    # Applies Gaussian noise to the value, centered at the value with a specified standard deviation.
    def _gaussian_noise(self, value):
        return random.gauss(value, self.standard_deviation)  # Use Gaussian distribution to add noise.


    # Applies noise based on Fractional Brownian Motion (FBM), adding a percentage of the current value based on the FBM sequence.
    def _FBM_noise(self, value):
        value = value + value * self.noise_sequence[self.number_of_increments]  # Modify the value based on the FBM noise sequence.
        self.number_of_increments += 1  # Move to the next value in the sequence.
        return value

    # Determines which noise function to apply based on the noise function type and applies it to the given value.
    def calculate_noisy_value(self, value):
        # Select the appropriate noise function based on the type specified and apply it to the input value.
        if self.noise_function_type == 'uniform_percentage':
            return self._uniform_percentage_based_noise(value)
        elif self.noise_function_type == 'uniform':
            return self._uniform_noise(value)
        elif self.noise_function_type == 'gaussian':
            return self._gaussian_noise(value)
        elif self.noise_function_type == 'FBM':
            return self._FBM_noise(value)
        else:
            print(f"No noise function found for type '{self.noise_function_type}'")
            return value  # Return the original value if no valid noise function type is found.


    # Generates a sequence simulating Fractional Brownian Motion (fBm) using the Davies-Harte method.
    def _FBM_sequence(self):
        '''
        Generates sample paths of fractional Brownian Motion using the Davies Harte method
        Forrás: Justin Yu https://github.com/732jhy/Fractional-Brownian-Motion
        '''
        N = self.length  # Number of increments.
        Hurst = self.hurst_parameter  # Hurst parameter, determines the memory of the process.
        gamma = lambda k, Hurst: 0.5 * (abs(k-1)**(2*Hurst) - 2*abs(k)**(2*Hurst) + abs(k+1)**(2*Hurst))
        g = [gamma(k, Hurst) for k in range(0, N)];  # Autocovariance function of the increments.
        r = g + [0] + g[::-1][0:N-1]  # Extended autocovariance function.

        # Eigenvalue decomposition step.
        j = np.arange(0, 2*N)
        k = 2*N - 1
        lk = np.fft.fft(r * np.exp(2 * np.pi * complex(0, 1) * k * j * (1 / (2 * N))))[::-1]  # Compute eigenvalues.

        # Generate random normal variables.
        Vj = np.zeros((2*N, 2), dtype=np.complex)
        Vj[0, 0] = np.random.standard_normal()
        Vj[N, 0] = np.random.standard_normal()
        for i in range(1, N):
            Vj1 = np.random.standard_normal()
            Vj2 = np.random.standard_normal()
            Vj[i][0] = Vj1; Vj[i][1] = Vj2
            Vj[2*N-i][0] = Vj1; Vj[2*N-i][1] = Vj2

        # Compute the Fourier transform to generate the fBm sample path.
        wk = np.zeros(2*N, dtype=np.complex)
        wk[0] = np.sqrt(lk[0] / (2 * N)) * Vj[0][0]
        wk[1:N] = np.sqrt(lk[1:N] / (4 * N)) * (Vj[1:N].T[0] + (complex(0, 1) * Vj[1:N].T[1]))
        wk[N] = np.sqrt(lk[0] / (2 * N)) * Vj[N][0]
        wk[N+1:2*N] = np.sqrt(lk[N+1:2*N] / (4 * N)) * (np.flip(Vj[1:N].T[0]) - (complex(0, 1) * np.flip(Vj[1:N].T[1])))

        Z = np.fft.fft(wk)  # Compute the Fourier transform of the weighted increments.
        fGn = Z[0:N]  # Fractional Gaussian noise.
        fBm = np.cumsum(fGn) * (N**(-Hurst))  # Convert to fractional Brownian motion.
        initial_value = 10  # Set initial value.
        fBm = np.array([initial_value] + list(fBm + initial_value))
        path = [x.real for x in fBm]  # Convert complex to real.
        percentage_path = [0] + [(path[i] - path[i-1]) / path[i-1] for i in range(1, len(path))]  # Calculate percentage changes.

        # Apply a transformation to ensure the percentages are bounded.
        percentage_path = np.tanh(percentage_path)
        # Scale the results based on the Hurst parameter.
        percentage_path = [x * (Hurst * 5)**2 for x in percentage_path]
        return percentage_path  # Return the percentage path for the fractional Brownian motion.






