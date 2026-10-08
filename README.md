# Stock_Prediction: Time-Series Analysis and Stock Market Prediction**

This project focuses on the research and application of Deep Learning models for time-series data analysis, specifically targeting price movement predictions for the Nasdaq and Vietnam stock markets.

**1. Introduction**
   
   The primary objective of this project is to develop a system capable of forecasting market trends and volatility over a 5-day horizon. The system integrates modern neural network architectures to optimize feature extraction and capture complex temporal dependencies.

**2. Dataset**
   
   For this project, I used two (02) datasets: the first is NASDAQ, containing international tickers collected from the NASDAQ stock market. The second one is Vietnam, containing Vietnam tickers, collected on the HOSE stock market.

   Please visit this Google Drive link for the dataset: https://drive.google.com/drive/folders/1py-9WatW9vAtXJCIDJCWONU-BGGWdAc2?usp=sharing


**3. Model Architecture**

The project implements a hybrid architecture consisting of:

1D Convolutional Neural Networks (1D CNN): Utilized for extracting local spatial features and patterns from the time-series sequences.

Bidirectional Gated Recurrent Units (BiGRU): Employed to learn long-term dependencies by processing data in both forward and backward time directions.

Attention Mechanism: Integrated to allow the model to focus on the most significant historical time steps that influence future price actions.

The system produces three distinct outputs:

Regression Model: Forecasts the specific future value of the stock indices.

Threshold Model: Predicts volatility levels based on predefined movement thresholds.

Classification Model: Determines the overall market direction (Up/Down).

**4. Repository Structure**

The repository is organized as follows:

Stock Prediction in Deep Learning.ipynb: The primary Jupyter notebook containing the end-to-end pipeline, including data preprocessing, model definition, training, and evaluation.

NASDAQ.rar: Contains the trained model weights (.h5 files) and configurations specifically optimized for the Nasdaq dataset.

VIETNAM.rar: Contains the trained model weights (.h5 files) and configurations specifically optimized for the Vietnam stock dataset.


**5. Evaluation Metrics**

The models are validated using standard performance indicators:

For Regression: Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).

For Classification: Accuracy, Precision, Recall, and F1-score.
