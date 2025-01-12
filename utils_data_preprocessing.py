import pandas as pd
import numpy as np

class Utils_data_preprocessing:
    def __init__(self):
        pass

    

    def clean_emdat(self, 
                    emdat_df: pd.DataFrame, 
                    minimum_start_year: int, 
                    accepted_disasters_types: list) -> pd.DataFrame:
        """
        Cleans and preprocesses an EM-DAT DataFrame

        Parameters:
            -emdat_df (pd.DataFrame): emdat data.
            -minimum_start_year (int): The earliest start year to include in the filtered data.
            -accepted_disasters_types (list): A list of disaster types to retain in the filtered data.

        Returns:
            pd.DataFrame: The cleaned and filtered DataFrame with formatted date columns 
                        and a calculated 'Duration Days' column.
        """

        # Drop rows with NAs in their dates
        clean_emdat_df = emdat_df.dropna(subset=['Start Year', 'Start Month', 'Start Day', 'End Year', 'End Month', 'End Day'])

        # Delete all the disasters that occured before 'minimum_start_year'
        clean_emdat_df = clean_emdat_df[clean_emdat_df["Start Year"] >= minimum_start_year]

        # Keep only the disasters types that we want
        clean_emdat_df = clean_emdat_df[clean_emdat_df["Disaster Type"].isin(accepted_disasters_types)]

        # Create complete dates in NOAA format, YYYY-MM-DD. Example: 2000-01-02
        # Create a new 'date' column by combining year, month, and day
        clean_emdat_df['Start Date'] = pd.to_datetime(clean_emdat_df[['Start Year', 'Start Month', 'Start Day']].astype(int).astype(str).agg('-'.join, axis=1))
        clean_emdat_df['End Date'] = pd.to_datetime(clean_emdat_df[['End Year', 'End Month', 'End Day']].astype(int).astype(str).agg('-'.join, axis=1))

        # create Duration Days
        clean_emdat_df['Duration Days'] = (clean_emdat_df['End Date'] - clean_emdat_df['Start Date']).dt.days #.dt.days to convert to int

        # Optional: format the date column to YYYY-MM-DD
        clean_emdat_df['Start Date'] = clean_emdat_df['Start Date'].dt.strftime('%Y-%m-%d')
        clean_emdat_df['End Date'] = clean_emdat_df['End Date'].dt.strftime('%Y-%m-%d')

        return clean_emdat_df


    def count_diasters_by_day(self, 
                              clean_emdat_df: pd.DataFrame, 
                              noaa_df: pd.DataFrame, 
                              n_after_disaster_days_to_label: int) -> pd.DataFrame:
        
        """
        Joins emdat and noaa dfs.
        Counts the number of disasters of each date and state from clean_emdat_df then creates a column called 'number_disasters'
        in the noaa_df with the corresponding count.

        Params: 
            -clean_emdat_df (pd.DataFrame): cleaned version of emdat df with the state column created
            -noaa_df (pd.DataFrame): noaa data
            -n_after_disaster_days_to_label (int): if the disaster lasts more than 'n_after_disaster_days_to_label' days, only the
                first 'n_after_disaster_days_to_label days are counted as disaster. 
                Example: if a disaster starts: 7th June and ends 10th June. If 'n_after_disaster_days_to_label' = 1, 7th June
                and 8th June will be labelled as disaster but 9th, 10th not. If you want to label all days you can
                use 'n_after_disaster_days_to_label' = 9999 or if you want only the start day, then use 'n_after_disaster_days_to_label' = 0

        Returns:
            pd.DataFrame: noaa df with 'number_disasters'
        """
        # Dictionary that will contain all the dates of the disaster per state, example: {'Texas': [2023-01-28, 2023-01-30,...], 'Kansas': [...], ...}
        dates_by_states = dict.fromkeys(clean_emdat_df['State'].unique(), None)

        # Function to get all dates between Start Date and Start Date + 'n_after_disaster_days_to_label'
        # IMPORTANT: if Start Date + 'n_after_disaster_days_to_label' is after End Date, only dates until End Date will be taken no more because the disaster already ended
        def extract_dates(row):
            start = pd.to_datetime(row['Start Date'])
            end = pd.to_datetime(row['End Date'])

            # Check if the difference between start and end dates is bigger than 'n_after_disaster_days_to_label'
            if (end - start).days > n_after_disaster_days_to_label:
                # Limit to 'n_after_disaster_days_to_label' days after Start Date
                end_limit = start + pd.Timedelta(days=n_after_disaster_days_to_label)

            # if Start Date + 'n_after_disaster_days_to_label' is after End Date
            else:
                # Use the actual End Date as limit
                end_limit = end
            
            # Generate the date range and format as 'YYYY-MM-DD'
            return pd.date_range(start=start, end=end_limit).strftime('%Y-%m-%d').tolist()

        # iterate trough all states and apply the previous function
        for state in dates_by_states:
            # keep only disasters in a state
            filtered_df = clean_emdat_df[clean_emdat_df["State"] == state].copy() # copy needed

            # Apply the function to each row and aggregate all dates in a single list
            all_dates = []
            filtered_df['Date Range'] = filtered_df.apply(lambda row: extract_dates(row), axis=1)

            # Flatten the list of lists into a single list of dates in 'YYYY-MM-DD' format
            for dates in filtered_df['Date Range']:
                all_dates.extend(dates)
            
            dates_by_states[state] = all_dates
            # if two dates are repeated inside a state, that means that two disasters happened that day

        # Include 'number_disasters' variable to noaa_df
        # Step 1: Convert dates_by_states dictionary to a DataFrame with counts
        # Flatten the dictionary to create a DataFrame for counting occurrences
        dates_list = [(state, date) for state, dates in dates_by_states.items() for date in dates]
        state_dates_df = pd.DataFrame(dates_list, columns=['state', 'DATE'])

        # Group by 'State' and 'Date' to get the count of occurrences per date
        state_date_counts = state_dates_df.groupby(['state', 'DATE']).size().reset_index(name='number_disasters')

        # Step 2: Merge the count DataFrame with the original DataFrame on 'State' and 'Date'
        noaa_counted = noaa_df.merge(state_date_counts, on=['state', 'DATE'], how='left')

        # Step 3: Fill NaN values with 0 if there was no match in the dictionary, because that means no disasters ocurred that day
        noaa_counted['number_disasters'] = noaa_counted['number_disasters'].fillna(0).astype(int)

        return noaa_counted


    def prepare_state_version_data_for_model(self, 
                                             data: pd.DataFrame, 
                                             selected_state: str, 
                                             n_next_days_until_disaster: int, 
                                             lengths_days_ma: list, 
                                             max_lag_period: int) -> pd.DataFrame:
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
            -data_grouped (pd.DataFrame): data ready to input to models
        """

        # 1.Aggregate Station Data
        data_grouped = data.groupby(['state', 'DATE']).agg({
        'VPDMAX': ['mean', 'max', 'min'],# 'std'], # NEW DATA COLUMN
        'VPDMIN': ['mean', 'max', 'min'],# 'std'], # NEW DATA COLUMN
        'TDMEAN': ['mean', 'max', 'min'],# 'std'], # NEW DATA COLUMN
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

        # 5. Sort chronologically, missing dates, then drop Date
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
        data_grouped.drop(columns=['DATE'], inplace=True) 

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
        data_grouped.drop(columns=["occured_disaster"], inplace=True)
        
        # 7. Moving averages
        columns_to_ma = data_grouped.drop(columns=
                              ['season_autumn','season_spring', 'season_summer', 'season_winter', 'target']
                              ).columns.to_list()

        # Create a dictionary to store all new ma columns
        ma_columns = {}
        for n_day in lengths_days_ma:
            for col in columns_to_ma:
                # Calculate lengths_days_ma-day moving average
                ma_columns[f'{col}_{n_day}_day_MA'] = data_grouped[col].rolling(window=n_day).mean()
        
        # Create a DataFrame from the ma columns dictionary
        ma_df = pd.DataFrame(ma_columns)
        # Concatenate the ma DataFrame with the original DataFrame
        data_grouped = pd.concat([data_grouped, ma_df], axis=1)

        # 8. Lagged variables
        columns_to_lag = columns_to_ma.copy() # no estoy laggeando los MA's pq probé y empeoraban el modelo????????

        # Create a dictionary to store all new lagged columns
        lagged_columns = {}
        # Loop to create lagged values and store them in the dictionary
        for col in columns_to_lag:
            for lag in range(1, max_lag_period + 1):
                lagged_columns[f'Lagged_{col}_{lag}'] = data_grouped[col].shift(lag)

        # Create a DataFrame from the lagged columns dictionary
        lagged_df = pd.DataFrame(lagged_columns)
        # Concatenate the lagged DataFrame with the original DataFrame
        data_grouped = pd.concat([data_grouped, lagged_df], axis=1)

        # 9.Important, drop the na's generated with lags, etc.
        data_grouped.dropna(inplace=True)

        return data_grouped


    def prepare_station_version_data_for_model(self, 
                                               data: pd.DataFrame, 
                                               selected_station: str, 
                                               n_next_days_until_disaster: int, 
                                               lengths_days_ma: list,
                                               max_lag_period: int) -> pd.DataFrame:
        
        """
        Preparates a dataframe so it can be used for training using ONE station. One-hot encoding categorical variables, filter by station,
        creates 'target' column, computes moving averages and lagged variables

        Params: 
            -data (pd.DataFrame): noaa with counted disasters column
            -selected_station (str): Sation to keep
            -n_next_days_until_disaster (int): when creating the target, we need to determine when to label a day before the disaster.
               If 'n_next_days_until_disaster' = 3, and the disaster ocurred the 4th May, then, 1st, 2nd, 3rd and 4th included will be labelled as 1
            -lengths_days_ma (list): a list with the length of the disired Moving Averages to compute. Example: 'lengths_days_ma'=[3, 7, 14] would 
                compute 3-day-MA, 7-day-MA and 14-day-MA
            -max_lag_period (int): the maximum number of lagged variables. Example: 'max_lag_period'=5, it will add the variables of the previous 5 days

        Returns:
            -data_station (pd.DataFrame): data ready to input to models
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

        return data_station


    def adjust_days_previous_disaster(self, 
                                      column_to_adjust: pd.Series, 
                                      n_next_days_until_disaster: int) -> pd.Series:
        # Creacion de la target: 
        # target = 1, si hoy hubo desastre o en los siguientes 7 días (incluyendo hoy) habrá desastre. 
        # Ejemplo: Si hubo un desastre durante el 30 de Enero hasta el 1 de Febrero. 
        # Entonces la target será = 1, los días 24,25,26,27,28,29,30,31,1.
        # El 23 NO es target=1 pq quedan 7 días siguidos enteros hasta el desastre (23,24,25,26,27,28,29)

        # Para ese ejemplo, la target del 23 de Enero al 1 de Febrero sería: 
        # [0,   1,     1,    1,    1,    1,    1, 1, 1, 1, 1] , si la ajustamos la convierte a
        # [0, 0.14, 0.29, 0.43, 0.57, 0.71, 0.86, 1, 1, 1, 1]

        # Processing logic
        consecutive_count = 0
        adjusted_column = []

        for value in column_to_adjust:
            if value == 1:
                consecutive_count += 1
                if consecutive_count < n_next_days_until_disaster:
                    # Replace the value based on its position in the sequence
                    adjusted_column.append(round(consecutive_count / n_next_days_until_disaster, 2))
                else:
                    adjusted_column.append(1)
            else:
                # Reset counter for consecutive 1s
                consecutive_count = 0
                adjusted_column.append(0)

        # the adjusted_column back
        return adjusted_column


    def prepare_state_version_data_for_model_predict(self, 
                                             data: pd.DataFrame, 
                                             selected_state: str, 
                                             n_next_days_until_disaster: int, 
                                             lengths_days_ma: list, 
                                             max_lag_period: int) -> pd.DataFrame:
        """
        JUST THE SAME AS THE OTHER FUNCTION BUT WITHOUT CLEANING NAN'S OF target COLUMN AT THE END
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
            -data_grouped (pd.DataFrame): data ready to input to models
        """

        # 1.Aggregate Station Data, if 3 station only have DATE until 20th of Oct, and the rest 2 have until 24th Oct, that days will appear and will be computed using only the stations that have data
        data_grouped = data.groupby(['state', 'DATE']).agg({
        'VPDMAX': ['mean', 'max', 'min'],# 'std'], # NEW DATA COLUMN
        'VPDMIN': ['mean', 'max', 'min'],# 'std'], # NEW DATA COLUMN
        'TDMEAN': ['mean', 'max', 'min'],# 'std'], # NEW DATA COLUMN
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

        # 5. Sort chronologically, missing dates, then drop Date
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
        data_grouped.drop(columns=['DATE'], inplace=True) 

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
        data_grouped.drop(columns=["occured_disaster"], inplace=True)
        
        # 7. Moving averages
        columns_to_ma = data_grouped.drop(columns=
                              ['season_autumn','season_spring', 'season_summer', 'season_winter', 'target']
                              ).columns.to_list()

        # Create a dictionary to store all new ma columns
        ma_columns = {}
        for n_day in lengths_days_ma:
            for col in columns_to_ma:
                # Calculate lengths_days_ma-day moving average
                ma_columns[f'{col}_{n_day}_day_MA'] = data_grouped[col].rolling(window=n_day).mean()
        
        # Create a DataFrame from the ma columns dictionary
        ma_df = pd.DataFrame(ma_columns)
        # Concatenate the ma DataFrame with the original DataFrame
        data_grouped = pd.concat([data_grouped, ma_df], axis=1)

        # 8. Lagged variables
        columns_to_lag = columns_to_ma.copy() # no estoy laggeando los MA's pq probé y empeoraban el modelo????????

        # Create a dictionary to store all new lagged columns
        lagged_columns = {}
        # Loop to create lagged values and store them in the dictionary
        for col in columns_to_lag:
            for lag in range(1, max_lag_period + 1):
                lagged_columns[f'Lagged_{col}_{lag}'] = data_grouped[col].shift(lag)

        # Create a DataFrame from the lagged columns dictionary
        lagged_df = pd.DataFrame(lagged_columns)
        # Concatenate the lagged DataFrame with the original DataFrame
        data_grouped = pd.concat([data_grouped, lagged_df], axis=1)

        # 9.Important, drop the na's generated with lags, etc. BUT EXCEPT target NANs
        data_grouped = data_grouped.dropna(subset=[col for col in data_grouped.columns if col != 'target'])

        return data_grouped

if __name__ == "__main__":
    print("Running utils_data_preprocessing.py directly")
