# Disaster Forecasting in the Southern United States

This repository provides a complete pipeline for forecasting the likelihood of climate-related disasters (floods and storms) occurring in Southern U.S. states up to 21 days in advance, using LSTM-based time series modeling.

---

## Table of Contents

* [Features](#features)
* [Repository Structure](#repository-structure)
* [Data Description](#data-description)
* [Installation](#installation)
* [Usage](#usage)

  * [1. Data Preprocessing](#1-data-preprocessing)
  * [2. Hyperparameter Tuning (Auto Pipeline)](#2-hyperparameter-tuning-auto-pipeline)
  * [3. Model Training](#3-model-training)
  * [4. Prediction](#4-prediction)
  * [5. Post-Inference Analysis](#5-post-inference-analysis)
* [Results](#results)
* [Contributing](#contributing)
* [License](#license)

---

## Features

* **State & Disaster-Specific Models**: Train separate LSTM classifiers for each U.S. state and disaster type (floods, storms), capturing localized patterns.
* **Flexible Prediction Window**: Forecast disaster occurrence probabilities up to 21 days ahead, with options to narrow the window (e.g., 5 days) for precision-critical applications.
* **Imbalanced Data Handling**: Apply custom class weighting to penalize false negatives twice as heavily as false positives, improving disaster detection recall.
* **Automated Hyperparameter Tuning**: Explore 8 LSTM hyperparameters and 7 data-processing parameters via an interactive Jupyter pipeline.
* **End-to-End Pipeline**: From raw EM-DAT and meteorological CSVs to final prediction outputs and evaluation metrics.

---

## Repository Structure

```bash
├── data/                          # Raw datasets
│   ├── emdat_ready.csv            # Historical disaster records from EM-DAT
│   └── meteostat_noaa.csv         # Daily climate data for selected stations
│
├── trained_models/                # Saved Keras `.keras` model files (per state & disaster)
│   └── *.keras
│
├── predictions/                   # Output CSV predictions
│   ├── <State>_<Event>_pred_prob_next_21_days.csv
│   └── full_all_predictions.csv   # Combined results
│
├── requirements.txt               # Python dependencies
├── utils_data_preprocessing.py    # Data cleaning & feature-engineering utilities
├── lstm_classifier.py             # LSTM model definition & training routines
├── post_inference.py              # Functions for preparing and analyzing inference data
├── auto-pipeline-lstm.ipynb       # Interactive hyperparameter tuning & model pipeline
├── predict_lstm.ipynb             # Prediction workflow using tuned parameters
└── README.md                      # Project overview (this file)
```

---

## Data Description

* **EM-DAT Data (`data/emdat_ready.csv`)**: Contains disaster event records (type, date, location) filtered to southern states since a configurable start year.
* **Meteorological Data (`data/meteostat_noaa.csv`)**: Daily climate measurements (temperature, precipitation, etc.) from five stations per state.

> **Note**: Result CSVs (`.csv`) and serialized models (`.keras`) are generated outputs. They are included for reference but need not be modified.

---

## Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/Climate-Disasters-Forecasting-Southern-USA.git
   cd Climate-Disasters-Forecasting-Southern-USA
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### 1. Data Preprocessing

Use `utils_data_preprocessing.py` to clean EM-DAT records, merge with meteorological data, and engineer features:

```python
from utils_data_preprocessing import Utils_data_preprocessing

prep = Utils_data_preprocessing()
emdat = prep.clean_emdat(emdat_df, minimum_start_year=2000, accepted_disasters_types=['Storm', 'Flood'])
# Merge with climate data, generate lags, moving averages, one-hot encoding, etc.
```

### 2. Hyperparameter Tuning (Auto Pipeline)

Explore LSTM architectures and data parameters interactively in `auto-pipeline-lstm.ipynb`. Adjust search space for:

* LSTM units, dropout rates, learning rate, regularization
* Window size, lag lengths, aggregation periods, class weights

### 3. Model Training

Train or retrain state-event models via `lstm_classifier.py`:

```python
from lstm_classifier import Lstm_classifier

clf = Lstm_classifier()
clf.train(X_train, y_train, hyperparameters)
clf.save_model('trained_models/Texas_Flood_pred_prob_next_21_days.keras')
```

### 4. Prediction

Generate future disaster probability forecasts using `predict_lstm.ipynb` or the CLI:

```bash
python post_inference.py --state Texas --event Flood --next_days 21
```

This will output CSV files in `predictions/`.

### 5. Post-Inference Analysis

Analyze model performance with `post_inference.py`:

```python
from post_inference import Post_inference

post = Post_inference()
metrics_df = post.evaluate_predictions(pred_csv, true_labels)
post.plot_metrics(metrics_df)
```

---

## Results

* **Evaluation Metrics**: F1-Score, Precision, Recall, Refined Recall.
* **Regional Insights**: Compare performance across states and disaster types.

![Model Performance by State](https://github.com/yourusername/assets/performance_by_state.png)

---

## Contributing

Contributions are welcome! Please open issues or submit pull requests for:

* Support for additional disaster types or regions
* Alternative model architectures (e.g., Transformer networks)
* Enhanced visualization dashboards

---

## License

This project is licensed under the MIT License. See `LICENSE` for details.
