<p align="right">
  # Disaster Prediction in the Southern United States <img src="https://github.com/user-attachments/assets/e717fc06-f2d4-4ca1-96b6-a78c4a24bd3c" alt="Image Description" width="200" />
</p>

This repository contains the code and configuration for predicting tornadoes, cyclones, and floods in the southern United States. The prediction models can forecast disasters between 21 and 5 days in advance, using LSTM

## Key Features

### Prediction Window
- The model predicts disasters up to **21 days in advance**, narrowing the window to as close as **5 days** for increased precision.

### Class Weights
- To emphasize the importance of correctly identifying disasters, misclassifying disasters is penalized **twice as much** compared to non-disasters.

### Localized Models
- Models are trained separately for each pair of **state** and **disaster type**, ensuring tailored predictions for specific regions and event types.

## Model Configurations
We offer three model configurations: **Small**, **Medium**, and **Large**, each designed to balance resource usage and prediction accuracy.

### 1. Small Model
- **Units**: 75
- **Learning Rate**: 1e-4
- **Epochs**: 5
- **Dropout**: 0.2
- **L2 Regularizer Weight**: 0.1

### 2. Medium Model (x2 of Small)
- **Units**: 150
- **Learning Rate**: 1e-5
- **Epochs**: 8
- **Dropout**: 0.2
- **L2 Regularizer Weight**: 0.01

### 3. Large Model (x6 of Small)
- **Units**: 450
- **Learning Rate**: 1e-6
- **Epochs**: 20
- **Dropout**: 0.2
- **L2 Regularizer Weight**: 0.001
- **Architecture**: Two LSTM layers with **8 hyperparameters** and **7 data parameters**.

## Dataset
The models are trained on historical data specific to the southern United States. The dataset includes:
- Historical records of tornadoes, cyclones, and floods.
- Meteorological data spanning temperature, humidity, wind speeds, precipitation, etc.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/eSVeeF/-Climate-Disasters-Forecasting-Southern-USA-.git
   cd -Climate-Disasters-Forecasting-Southern-USA-
   ```
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Results
The prediction performance metrics include:
- F1-Score, Precision, Recall, and Refined Recall for disaster prediction.
- Region-specific accuracy analysis.

  
![image](https://github.com/user-attachments/assets/6ef87921-e2c3-4ad9-8a32-e454b2cbe620)
![image](https://github.com/user-attachments/assets/e9055d1e-06d7-4dd1-93f7-f06bc702538a)
