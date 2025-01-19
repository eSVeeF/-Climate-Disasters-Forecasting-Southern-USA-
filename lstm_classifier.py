import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
# from sklearn.model_selection import TimeSeriesSplit     cross-validation
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt


class Lstm_classifier:
    def __init__(self):
        pass

    
    def scale_data(self,
                   data: pd.DataFrame,
                   choosen_scaler: str, 
                   cols_not_scale: list) -> np.ndarray:
        
        """
        Scales specified columns of a DataFrame while excluding certain columns from scaling. 
        But the not scaled_cols are included in the return (they are not removed, just not scaled)

        Parameters:
        - data (pd.DataFrame): The input DataFrame to scale.
        - choosen_scaler (str): The scaler to use ('standard', 'minmax', or 'quantile').
        - cols_not_scale (list): List of column names to exclude from scaling.

        Returns:
        - np.ndarray: A NumPy array containing the scaled data, with excluded columns concatenated.
        """
        data_to_scale = data.drop(columns=cols_not_scale)

        # Options for scalers, qunatiles need random_state so it can replicate conditions
        scalers_dict = {'standard': StandardScaler(), 'minmax': MinMaxScaler(), 'quantile': QuantileTransformer(random_state=42)}
        
        scaler = scalers_dict[choosen_scaler.lower()]

        # Apply scaling only to the selected columns
        scaled_data = scaler.fit_transform(data_to_scale)

        # Concatenate the scaled columns with the unscaled ones
        scaled_data = np.concatenate([scaled_data, data[cols_not_scale].drop(columns='target')], axis=1)
        return scaled_data
    
    def create_sequences(self, 
                         features,
                         target, 
                         sequence_length):
        X, y = [], []
        start_idx = len(features) % sequence_length  # Skip the first few points to ensure last data points are included
        for i in range(start_idx, len(features) - sequence_length):
            X.append(features[i:i + sequence_length])
            y.append(target.iloc[i + sequence_length])  # Use .iloc for pd.Series)
        return np.array(X), np.array(y)

    def lstm_time_series_train_test_split(self, 
                                          scaled_data: np.ndarray,
                                          target_column: pd.Series,  
                                          sequence_length: int, 
                                          train_test_split_ratio: float):
        
        """
        Splits time-series data into training and testing sets for LSTM modeling.

        Params:
            -scaled_data (np.ndarray): Scaled time-series data for feature generation.
            -target_column (pd.Series): Column representing target values for prediction.
            -sequence_length (int): It specifies the length of the input sequence the 
                lstm model will use to make predictions. For example, if sequence_length = 7, 
                the model will use data from the past 7 time steps (e.g., 7 days) to predict 
                the target value for the next time step.
            -train_test_split_ratio (float): Proportion of data to allocate to the training set 
                                            (e.g., 0.8 for 80% training data).

        Returns:
            A tuple containing:
                - X_train (np.ndarray): Training set input sequences.
                - X_test (np.ndarray): Testing set input sequences.
                - y_train (np.ndarray): Training set target values.
                - y_test (np.ndarray): Testing set target values.
        """
        # 1.Create sequences for LSTM (sequence_length-day input sequence to predict disaster probability in the next 'sequence_length' days)
        X, y = self.create_sequences(scaled_data, target_column, sequence_length)

        # 2.Train-test split, for time-series
        n_total_x = len(X)
        train_size_x = int(n_total_x * train_test_split_ratio)  
        X_train = X[:train_size_x]
        X_test = X[train_size_x:]

        n_total_y = len(y)
        train_size_y = int(n_total_y * train_test_split_ratio) 
        y_train = y[:train_size_y]
        y_test = y[train_size_y: ]

        return X_train, X_test, y_train, y_test
    
    def train_lstm(self, 
                   X_train: np.ndarray, 
                   y_train: np.ndarray, 
                   units: int, 
                   dropout: float, 
                   l2_regularizer_weight: float,
                   learning_rate: float, 
                   class_1_weight: float, 
                   epochs: int, 
                   batch_size: int, 
                   validation_data: tuple,
                   verbose: int,
                   show_plots: bool):
        
        """
        Trains an LSTM model for binary classification.

        Params:
        - X_train (np.ndarray): Training data, without validation set
        - y_train (np.ndarray): Training labels, without validation set
        - units (int): Number of units in the LSTM layers.
        - dropout (float): Dropout rate for regularization (between 0 and 1).
        - l2_regularizer_weight (float): Weight for l2 regularization
        - learning_rate (float): Learning rate for the Adam optimizer.
        - class_1_weight (float): Weight for class 1 in the class weighting scheme to handle imbalance.
        - epochs (int): Number of epochs for training.
        - batch_size (int): Size of the mini-batches during training.
        - validation_data (tuple): (X_val, y_val).
        - verbose (int): how much info is showed when training, 0=Silent, 1=default, 2=detailed.
        - show_plots (bool): Whether to display training/validation loss, precision, and recall plots.

        Returns:
        - model (tf.keras.Model): The trained LSTM model.
        """

        # 1. Build LSTM model
        model = Sequential()
        model.add(Input(shape=(X_train.shape[1], X_train.shape[2])))
        model.add(LSTM(units=units, return_sequences=True, kernel_regularizer=l2(l2_regularizer_weight)))
        model.add(Dropout(dropout))
        model.add(LSTM(units=units//2, return_sequences=False, kernel_regularizer=l2(0.01))) # With return_sequences=True in the final LSTM layer, the output is a 3D tensor with a shape of (batch_size, sequence_length, 1)
        model.add(Dropout(dropout))
        model.add(Dense(1, activation='sigmoid'))  # Sigmoid activation for binary classification, if we use adjusted data maybe not necesary to use sigmoid

        # 2.Compile model
        model.compile(optimizer=Adam(learning_rate=learning_rate, clipnorm=1.0), loss='binary_crossentropy', metrics=[Precision(), Recall()])  

        # 3.Train model
        class_weights = {0: 1, 1: class_1_weight} # Make more penalty when missing disasters (High imbalanced problem), usar esto sum(y_train == 0)/sum(y_train == 1)
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=validation_data, class_weight=class_weights, verbose=verbose)

        # Plots
        if show_plots:
            # Plot training loss and validation loss
            plt.plot(history.history['loss'], label='Training Loss')
            plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.title('Model Loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend()
            plt.show()

            # Plot precison, recall, for training and validation
            plt.plot(history.history['precision'], label='Training Precision')
            plt.plot(history.history['recall'], label='Training Recall')
            plt.plot(history.history['val_precision'], label='Validation Precision')
            plt.plot(history.history['val_recall'], label='Validation Recall')
            plt.title('Model Precision and Recall')
            plt.ylabel('Precision and Recall')
            plt.xlabel('Epoch')
            plt.legend()
            plt.show()

        return model
    
    def evaluate_lstm(self, 
                      model, 
                      X_test: np.ndarray, 
                      y_test: np.ndarray, 
                      is_target_adjusted: bool,
                      verbose: int,
                      show_plot: bool):
        
        """
        Evaluates an LSTM model on a test set and determines the optimal thresholds 
        for precision, recall, and F1-score.

        Parameters:
        - model: The trained LSTM model to evaluate.
        - X_test (np.ndarray): Training data, from 'lstm_time_series_train_test_split()'.
        - y_test (np.ndarray): Training labels, from 'lstm_time_series_train_test_split()'.
        - is_target_adjusted (bool): Whether if the data was adjusted or not using 'Utils_data_preprocessing().adjust_days_previous_disaster()'.
        - verbose (int): if = 0, no info will be printed
        - show_plot (bool)

        Returns:
        - probs_predictions (np.ndarray): Predicted probabilities for the test set.
        - best_f1_threshold (float): Threshold for the highest F1-score.
        - metrics_best_f1_threshold (tuple): (F1-score, Precision, Recall) for the threshold with best F1 score 
        - metrics_05_threshold (tuple same but for threshold 0.5)
        """

        loss, precison, recall = model.evaluate(X_test, y_test, verbose=verbose)
        if verbose > 0:
            print(f'Test Loss: {loss}, Test Precison: {precison}, Test Recall: {recall}')
        probs_predictions = model.predict(X_test).flatten()

        if is_target_adjusted:
            y_test = [1 if i > 0 else 0 for i in y_test]

        # Select the best threshold (default is 0.5)
        y_probs = probs_predictions  # Probabilities between 0 and 1
        y_true = y_test.copy()   # True labels (0 or 1)

        # Initialize variables to store the best metrics and corresponding thresholds
        best_f1 = 0
        best_f1_threshold = 0
        best_f1_pre = 0
        best_f1_recall = 0
        precision_scores = []
        recall_scores = []
        f1_scores = []

        # Evaluate precision, recall, and F1-score at different thresholds
        thresholds = [i * 0.001 for i in range(1, 1000)]
        for threshold in thresholds:
            # Convert probabilities to binary predictions based on the current threshold
            y_pred = [1 if prob >= threshold else 0 for prob in y_probs]
            
            # Calculate precision, recall, and F1-score
            # if precision (tp/tp+fp) has zero_division, that is that there where no tp and no fp, so precision should not be taken into account later
            precision = precision_score(y_true, y_pred, zero_division=0)
            # if recall (tp/tp+fn) has zero_division, that is that there where no tp and no fn, so recall should not be taken into account later
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)

            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1)
            
            # Check and store the best F1 thresholds
            if f1 > best_f1:
                best_f1 = f1
                best_f1_threshold = threshold
                best_f1_pre = precision
                best_f1_recall = recall

        metrics_best_f1_threshold=(best_f1, best_f1_pre, best_f1_recall)

        # for threshold 0.5
        y_pred_05 = [1 if prob >= 0.5 else 0 for prob in y_probs]
        metrics_05_threshold = (f1_score(y_true, y_pred_05, zero_division=0), precision_score(y_true, y_pred_05, zero_division=0) , recall_score(y_true, y_pred_05, zero_division=0))

        if verbose > 0:
            # Output the results
            print(f"Best F1 Score: {best_f1:.3f} at Threshold: {best_f1_threshold}. Corresponding: Precision: {best_f1_pre} and Recall: {best_f1_recall}")
            print(f"Metrics at threshold = 0.5: F1-score: {metrics_05_threshold[0]}: Precision: {metrics_05_threshold[1]} and Recall: {metrics_05_threshold[2]}")

        # Plots
        if show_plot:
            # Plot precision and recall vs. thresholds
            plt.figure(figsize=(8, 5))
            plt.plot(thresholds, precision_scores, label="Precision", color='blue')
            plt.plot(thresholds, recall_scores, label="Recall", color='orange')
            plt.plot(thresholds, f1_scores, label="F1-score", color='green')
            plt.xlabel("Threshold")
            plt.ylabel("Score")
            plt.title("Precision, Recall and F1 vs. Threshold")
            plt.legend(loc="best")
            xticks = np.arange(0.0, 1.05, 0.1)
            plt.xticks(xticks)
            plt.grid(which='both', linestyle='--', linewidth=0.3, color='gray')
            plt.show()

        return probs_predictions, best_f1_threshold, metrics_best_f1_threshold, metrics_05_threshold


        

if __name__ == "__main__":
    print("Running lstm_classifier.py directly")
