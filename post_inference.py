import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Post_inference:
    def __init__(self):
        pass

    def prepare_state_version_data_for_inference(self, 
                                             data: pd.DataFrame, 
                                             selected_state: str, 
                                             n_next_days_until_disaster: int) -> pd.DataFrame:
        """
        Preparates a dataframe so it can be used for training using ONE state. Aggregates by stations, one-hot encoding categorical variables, filter by state,
        creates 'target' column, computes moving averages and lagged variables

        Params: 
            -data (pd.DataFrame): noaa with counted disasters column
            -selected_state (str): State to keep
            -n_next_days_until_disaster (int): when creating the target, we need to determine when to label a day before the disaster.
               If 'n_next_days_until_disaster' = 3, and the disaster ocurred the 4th May, then, 1st, 2nd, 3rd and 4th included will be labelled as 1
            -lengths_days_ma (list): a list with the length of the disired Moving Averages to compute. Example: 'lengths_days_ma'=[3, 7, 14] would 
                compute 3-day-MA, 7-day-MA and 14-day-MA
            -max_lag_period (int): the maximum number of lagged variables. Example: 'max_lag_period'=5, it will add the variables of the previous 5 days

        Returns:
            pd.DataFrame: data ready to input to models
        """
        # 1.Aggregate Station Data
        data_grouped = data.groupby(['state', 'DATE']).agg({
        'TAVG': ['mean', 'max', 'min'],# 'std'], # NEW DATA COLUMN
        'PRES': ['mean', 'max', 'min'],# 'std'], # NEW DATA COLUMN
        'AWND': ['mean', 'max', 'min'],# 'std'],
        'PRCP': ['mean', 'max', 'min'],# 'std'],
        'SNOW': ['mean', 'max', 'min'],# 'std'],
        'SNWD': ['mean', 'max', 'min'],# 'std'],
        'TMAX': ['mean', 'max', 'min'],# 'std'],
        'TMIN': ['mean', 'max', 'min'],# 'std'],
        'WDF2': ['mean', 'max', 'min'],
        'WDF5': ['mean', 'max', 'min'],
        'WSF2': ['mean', 'max', 'min'],
        'WSF5': ['mean', 'max', 'min'],
        'WT02': ['mean'], # todas las WT's son binarias entonces la mean mira cuantas estaciones han tenido ese Weather Type
        'WT01': ['mean'],
        'WT03': ['mean'],
        'WT04': ['mean'],
        'WT05': ['mean'],
        'WT06': ['mean'],
        'WT07': ['mean'],
        'WT08': ['mean'],
        'WT09': ['mean'],
        'WT10': ['mean'],
        'WT11': ['mean'],
        'WT13': ['mean'],
        'WT16': ['mean'],
        'WT17': ['mean'],
        'WT18': ['mean'],
        'WT19': ['mean'],
        'WT22': ['mean'],
        'season': lambda x: x.mode().iloc[0], # este mode da igual pq los grupos van a tener el mismo valor, The mode() method returns a Series, so iloc[0] is used to select the first mode in case there are multiple modes.
        'number_disasters': 'max'  # number_disasters for each day
        }).reset_index()

        # Flatten MultiIndex columns created by aggregation
        data_grouped.columns = ['_'.join(col).strip() if type(col) is tuple else col for col in data_grouped.columns]
        # fix one error
        data_grouped.rename(columns={'DATE_': 'DATE', 'state_':'state', 'season_<lambda>': 'season'}, inplace=True)
        
        # 2. Create binary version of 'number_disasters'
        data_grouped['occured_disaster'] = data_grouped["number_disasters_max"].apply(lambda x: 1 if x > 1 else x)
        data_grouped.drop(columns=["number_disasters_max"], inplace=True)

        # 3. one-hot endode categorical variables
        data_grouped = pd.get_dummies(data_grouped, columns=['season', 'state'], dtype=float)
        
        # 4. Filter by States
        # Filter
        state_column = f"state_{selected_state}"
        data_grouped = data_grouped[data_grouped[state_column]==1]
        
        # Drop the 'state_...' columns
        columns_to_drop = [col for col in data_grouped.columns 
                            if col.startswith('state_')]
        data_grouped.drop(columns=columns_to_drop, inplace=True)

        # 5. Sort chronologically, missing dates
        # Convert 'date' column to datetime
        data_grouped['DATE'] = pd.to_datetime(data_grouped['DATE'])
        # Generate complete date range
        full_range = pd.date_range(start=data_grouped['DATE'].min(), end=data_grouped['DATE'].max())
        # Identify missing dates
        missing_dates = full_range.difference(data_grouped['DATE'].dropna())
        # Print missing dates
        if len(missing_dates) > 0:
            print("CAUTION!!! There are missing dates:")
            for date in missing_dates:
                print(date)

        data_grouped = data_grouped.sort_values('DATE').reset_index(drop=True)
        data_grouped.dropna(inplace=True)

        # 6. Create the target
        # Creacion de la target: 
        # target = 1, si hoy hubo desastre o en los siguientes 7 días (incluyendo hoy) habrá desastre. 
        # Ejemplo: Si hubo un desastre durante el 30 de Enero hasta el 1 de Febrero. 
        # Entonces la target será = 1, los días 24,25,26,27,28,29,30,31,1. 
        # El 23 NO es target=1 pq quedan 7 días siguidos enteros hasta el desastre (23,24,25,26,27,28,29)

        # .rolling(window=7).max(): This rolling operation calculates the maximum value in each 7-day window (or 5-day, 3-day). 
        # For any 7-day window, if there is at least one 1 (disaster day), the result will be 1; otherwise, it will be 0.
        # .shift(-6): After calculating the rolling maximum for the next 7 days, we shift this result back by 6 days.
        # This way, the target label corresponds to the current day’s forecast of disasters within the next 7 days.
        data_grouped['target'] = data_grouped['occured_disaster'].rolling(window=n_next_days_until_disaster).max().shift(-(n_next_days_until_disaster-1))

        # Convert the 'timestamp' column to datetime format
        data_grouped['DATE'] = pd.to_datetime(data_grouped['DATE'])
        # Format the 'timestamp' column to 'YYYY-MM-DD'
        data_grouped['DATE'] = data_grouped['DATE'].dt.strftime('%Y-%m-%d')

        return data_grouped

    def prepare_station_version_data_for_inference(self, 
                                               data: pd.DataFrame, 
                                               selected_station: str, 
                                               n_next_days_until_disaster: int, 
                                               lengths_days_ma: list,
                                               max_lag_period: int) -> pd.DataFrame:
        
        """
        Preparates a dataframe so it can be used for inference  using ONE station

        Params: 
            -data (pd.DataFrame): noaa with counted disasters column
            -selected_station (str): Sation to keep
            -n_next_days_until_disaster (int): when creating the target, we need to determine when to label a day before the disaster.
               If 'n_next_days_until_disaster' = 3, and the disaster ocurred the 4th May, then, 2nd, 3rd and 4th included will be labelled as 1
            -lengths_days_ma (list): a list with the length of the disired Moving Averages to compute. Example: 'lengths_days_ma'=[3, 7, 14] would 
                compute 3-day-MA, 7-day-MA and 14-day-MA
            -max_lag_period (int): the maximum number of lagged variables. Example: 'max_lag_period'=5, it will add the variables of the previous 5 days

        Returns:
            pd.DataFrame: data ready to inference
        """
        
        # 1. No need of agrupation, because there is only one station
        data_station = data[data["NAME"]==selected_station]

        # Drop unecesary cols
        data_station = data_station.drop(columns=['NAME', 'STATION', 'state'])

        # 2. Create binary version of 'number_disasters'
        data_station['occured_disaster'] = data_station["number_disasters"].apply(lambda x: 1 if x > 1 else x)
        data_station = data_station.drop(columns=["number_disasters"])
        
        # 3. one-hot endode categorical variables
        data_station = pd.get_dummies(data_station, columns=['season'], dtype=float)

        # 4. Sort chronologically, missing dates, then drop Date
        # Convert 'date' column to datetime
        data_station['DATE'] = pd.to_datetime(data_station['DATE'])
        # Generate complete date range
        full_range = pd.date_range(start=data_station['DATE'].min(), end=data_station['DATE'].max())
        # Identify missing dates
        missing_dates = full_range.difference(data_station['DATE'].dropna())
        # Print missing dates
        if len(missing_dates) > 0:
            print("CAUTION!!! There are missing dates:")
            for date in missing_dates:
                print(date)

        data_station = data_station.sort_values('DATE').reset_index(drop=True)
        data_station.dropna(inplace=True)
        data_station.drop(columns=['DATE'], inplace=True) 

        # 5. Create the target
        # Creacion de la target: 
        # target = 1, si hoy hubo desastre o en los siguientes 7 días (incluyendo hoy) habrá desastre. 
        # Ejemplo: Si hubo un desastre durante el 30 de Enero hasta el 1 de Febrero. 
        # Entonces la target será = 1, los días 24,25,26,27,28,29,30,31,1. 
        # El 23 NO es target=1 pq quedan 7 días siguidos enteros hasta el desastre (23,24,25,26,27,28,29)

        # .rolling(window=7).max(): This rolling operation calculates the maximum value in each 7-day window (or 5-day, 3-day). 
        # For any 7-day window, if there is at least one 1 (disaster day), the result will be 1; otherwise, it will be 0.
        # .shift(-6): After calculating the rolling maximum for the next 7 days, we shift this result back by 6 days.
        # This way, the target label corresponds to the current day’s forecast of disasters within the next 7 days.
        data_station['target'] = data_station['occured_disaster'].rolling(window=n_next_days_until_disaster).max().shift(-(n_next_days_until_disaster-1))
        data_station.drop(columns=["occured_disaster"], inplace=True)

        # 6. Moving averages
        columns_to_ma = data_station.drop(columns=
                              ['season_autumn','season_spring', 'season_summer', 'season_winter', 'target']
                              ).columns.to_list()
        
        columns_to_ma = [item for item in columns_to_ma if not item.startswith('WT')]


        # Create a dictionary to store all new ma columns
        ma_columns = {}
        for n_day in lengths_days_ma:
            for col in columns_to_ma:
                # Calculate lengths_days_ma-day moving average
                ma_columns[f'{col}_{n_day}_day_MA'] = data_station[col].rolling(window=n_day).mean()
        
        # Create a DataFrame from the ma columns dictionary
        ma_df = pd.DataFrame(ma_columns)
        # Concatenate the ma DataFrame with the original DataFrame
        data_station = pd.concat([data_station, ma_df], axis=1)

        # 7. Lagged variables
        columns_to_lag = columns_to_ma.copy() # no estoy laggeando los MA's pq probé y empeoraban el modelo????????

        # Create a dictionary to store all new lagged columns
        lagged_columns = {}
        # Loop to create lagged values and store them in the dictionary
        for col in columns_to_lag:
            for lag in range(1, max_lag_period + 1):
                lagged_columns[f'Lagged_{col}_{lag}'] = data_station[col].shift(lag)

        # Create a DataFrame from the lagged columns dictionary
        lagged_df = pd.DataFrame(lagged_columns)
        # Concatenate the lagged DataFrame with the original DataFrame
        data_station = pd.concat([data_station, lagged_df], axis=1)

        # 9.Important, drop the na's generated with lags, etc.
        data_station.dropna(inplace=True)

        # Convert the 'timestamp' column to datetime format
        data_station['DATE'] = pd.to_datetime(data_station['DATE'])
        # Format the 'timestamp' column to 'YYYY-MM-DD'
        data_station['DATE'] = data_station['DATE'].dt.strftime('%Y-%m-%d')

        return data_station

    def inference_create_sequences(self, 
                                    inference_noaa_date_column,
                                    inference_noaa_occured_disaster_column, 
                                    sequence_length):
        X, y = [], []
        start_idx = len(inference_noaa_date_column) % sequence_length  # Skip the first few points to ensure last data points are included
        for i in range(start_idx, len(inference_noaa_date_column) - sequence_length):
            X.append(inference_noaa_date_column.iloc[i + sequence_length])
            y.append(inference_noaa_occured_disaster_column.iloc[i + sequence_length])
        return np.array(X), np.array(y)

    def inference_lstm_time_series_train_test_split(self, 
                                          inference_noaa_date_column: pd.Series,
                                          inference_noaa_occured_disaster_column: pd.Series,  
                                          sequence_length: int, 
                                          train_test_split_ratio: float):
        
        """
        Splits time-series data into training and testing sets for LSTM modeling.

        Params:
            -inference_noaa_date_column (pd.Series)
            -inference_noaa_occured_disaster_column (pd.Series)
            -sequence_length (int): It specifies the length of the input sequence the 
                lstm model will use to make predictions. For example, if sequence_length = 7, 
                the model will use data from the past 7 time steps (e.g., 7 days) to predict 
                the target value for the next time step.
            -train_test_split_ratio (float): Proportion of data to allocate to the training set 
                                            (e.g., 0.8 for 80% training data).

        Returns:
            A tuple containing:
                - inference_noaa_X_train (np.ndarray): Training set input sequences.
                - inference_noaa_X_test (np.ndarray): Testing set input sequences.
                - inference_noaa_y_train (np.ndarray): Training set target values.
                - inference_noaa_y_test (np.ndarray): Testing set target values.
        """
        # 1.Create sequences for LSTM (sequence_length-day input sequence to predict disaster probability in the next 'sequence_length' days)
        X, y = self.inference_create_sequences(inference_noaa_date_column, inference_noaa_occured_disaster_column, sequence_length)

        # 2.Train-test split, for time-series
        n_total_x = len(X)
        train_size_x = int(n_total_x * train_test_split_ratio)  
        inference_noaa_X_train = X[:train_size_x]
        inference_noaa_X_test = X[train_size_x:]

        n_total_y = len(y)
        train_size_y = int(n_total_y * train_test_split_ratio) 
        inference_noaa_y_train = y[:train_size_y]
        inference_noaa_y_test = y[train_size_y: ]

        return inference_noaa_X_train, inference_noaa_X_test, inference_noaa_y_train, inference_noaa_y_test
    
    def create_post_inference_data(self,
                                   inference_noaa_X_test: np.ndarray,
                                   inference_noaa_y_test: np.ndarray,
                                   y_test: np.ndarray,
                                   probs_predictions: np.ndarray,
                                   selected_threshold: float=0.5) -> pd.DataFrame:
        
        """
        Create a DataFrame containing post-inference data for disaster prediction analysis.

        Params:
            inference_noaa_X_test (np.ndarray): Array from inference_lstm_time_series_train_test_split().
            inference_noaa_y_test (np.ndarray): Array from inference_lstm_time_series_train_test_split().
            y_test (np.ndarray): Array from Lstm_classifier().lstm_time_series_train_test_split().
            probs_predictions (np.ndarray): Array from Lstm_classifier().evaluate_lstm().
            selected_threshold (float): the threshold used for prediciting 1 or 0 based on the probs, defalut=0.5

        Returns:
            pd.DataFrame: A DataFrame containing:
                - 'DATE': Corresponding dates from the test set.
                - 'occured_disaster': Ground truth for whether a disaster occurred on each date.
                - 'target': occured_disaster + prior days also labelled as 1's.
                - 'prediction_prob': Predicted probability of a disaster occurring.
                - 'prediction': Binary predictions (1 for disaster, 0 otherwise) based on the specified threshold.
        """

        post_inf=pd.DataFrame({'DATE': inference_noaa_X_test, 'occured_disaster': inference_noaa_y_test, 'target': y_test,  'prediction_prob': probs_predictions.flatten()})
        # Reset the index 
        post_inf.reset_index(drop=True, inplace=True)
        post_inf['prediction'] = post_inf['prediction_prob'].apply(lambda x: 1 if x>selected_threshold else 0)

        return post_inf

    def plot_probs(self,
                  minimum_days_prior: int,
                  post_inf: pd.DataFrame, 
                  n_next_days_until_disaster: int,
                  plot_width: int,
                  plot_height: int):
        
        """Creates a plot with the model's predicted diaster's probabilities + the actual disasters + the window of detection


        Params:
            -minimum_days_prior (int): number of days to plot in the prior window
            -post_inf (pd.DataFrame): df created using create_post_inference_data()
            -n_next_days_until_disaster (int): the same value used in the Utils_data_preprocessing().count_diasters_by_day()
        """

        maximum_days_prior = (n_next_days_until_disaster+1) 
        disaster_indices = post_inf.index[post_inf['occured_disaster'] == 1].tolist()

        # Indices where you want to add vertical lines 
        disaster_indices = post_inf.index[post_inf['occured_disaster'] == 1].tolist()

        # Create the line plot
        plt.figure(figsize=(plot_width, plot_height))
        plt.plot(post_inf['DATE'], post_inf['prediction_prob'], linestyle='-', color='b')
        plt.title('Date vs Predicted Prob of Disaster')
        plt.xlabel('Date')
        plt.ylabel('Predicted Prob of Disaster')
        plt.grid(True)
        # Add vertical lines at specified indices 
        for idx in disaster_indices: 
            plt.axvline(x=post_inf['DATE'][idx], color='r', linestyle='--')
            # Add a transparent rectangle, window of detection
            start_idx = max(idx - maximum_days_prior, 0)  # Ensure start index is within bounds
            end_idx = max(idx - minimum_days_prior, 0)  # Ensure end index is within bounds
            plt.axvspan(post_inf['DATE'][start_idx], post_inf['DATE'][end_idx], color='green', alpha=0.2)

        plt.axhline(y=0.5, color='purple', linestyle='-', linewidth=2, alpha=0.4)
        plt.ylim(0, 1)
        plt.xticks(rotation=90)
        plt.legend([f'Prediction Probability in the next {maximum_days_prior} days', 'Disaster', f'Prediction days interval ({maximum_days_prior} - {minimum_days_prior} pior days)'], loc='upper right', bbox_to_anchor=(0.99, 0.95), fontsize=9)
        plt.show()

    def refine_recall(self,
                      minimum_days_prior: int,
                      n_next_days_until_disaster: int,
                      post_inf: pd.DataFrame):
        
        """Refined recall: instead of check all 1's, check only if we have predicted at least one 1 before maximum_days_prior and minumum_days_prior the disaster,
        if there is at least one 1, the disaster is corretly predicted
        
        Params:
            -minimum_days_prior (int): number of days to plot in the prior window
            -post_inf (pd.DataFrame): df created using create_post_inference_data()
            -n_next_days_until_disaster (int): the same value used in the Utils_data_preprocessing().count_diasters_by_day()
            
        Returns: 
            refined_recall (float)"""
        
        maximum_days_prior = (n_next_days_until_disaster+1)
        # get the disaster indices
        disaster_indices = post_inf.index[post_inf['occured_disaster'] == 1].tolist()

        tp = 0
        fn = 0
        for idx in disaster_indices:
            # maximum_days_prior and minimum_days_prior window
            start_idx = max(idx - maximum_days_prior, 0)
            end_idx = max(idx - minimum_days_prior, 0)

            # check if at least there is one 1
            if post_inf['prediction'].iloc[start_idx:end_idx + 1].eq(1).any():
                tp += 1
            else:
                fn += 1

        # Calculate refined_recall
        refined_recall = tp / (tp + fn) if (tp + fn) != 0 else 0

        print(f'Number of disasters in this period {len(disaster_indices)}')
        print(f'refined_recall when predicting between {maximum_days_prior} - {minimum_days_prior} days previous a disaster: {refined_recall:.3f}')

        return len(disaster_indices), refined_recall
    
if __name__ == "__main__":
    print("Running post_inference.py directly")
