  # Disaster Prediction in the Southern United States 
  
This repository contains the code and configuration for predicting tornadoes, cyclones, and floods in the southern United States. The prediction models can forecast disasters between 21 and 5 days in advance, using LSTM with **8 hyperparameters** and **7 data parameters**

## Key Features

### Prediction Window
- The model predicts disasters up to **21 days in advance**, narrowing the window to as close as **5 days** for increased precision.

### Class Weights
- To emphasize the importance of correctly identifying disasters, misclassifying disasters is penalized **twice as much** compared to non-disasters.

### Localized Models
- Models are trained separately for each pair of **state** and **disaster type**, ensuring tailored predictions for specific regions and event types.

## Model Configurations
We offer three model configurations: **Small**, **Medium**, and **Large**, each designed to balance resource usage and prediction accuracy.

  <div style="display: flex; justify-content: space-between; align-items: center;">
  <img src="https://github.com/user-attachments/assets/f5475f6f-0bbf-407f-8630-aeffadfd9004" alt="Image 1 Description" width="600" />
  <img src="https://github.com/user-attachments/assets/e0e39954-190e-49aa-a077-4150c1f965a7" alt="Image 2 Description" width="604" />
  </div>


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
      
![image](https://github.com/user-attachments/assets/d229a2d8-1428-4f20-8edc-833808351cea)
![image](https://github.com/user-attachments/assets/ff3ca5be-553e-45cb-8bc0-acc9a2cc2c49)
