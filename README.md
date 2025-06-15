# Flood-Forecasting-Odisha
A deep learning-based flood forecasting system for the Mahanadi River Basin, Odisha, India.


# NeerDrishti Project Description
This project presents a deep learning-based flood forecasting system tailored specifically for the Mahanadi River Basin in Odisha, India. Frequent flooding in this region causes significant loss of life, damage to infrastructure, and economic disruption. Traditional flood forecasting models often fail to capture the complex nonlinear relationships and temporal dependencies required for accurate flood prediction, especially in regions with limited real-time sensor data.

To address these limitations, this project employs a Bidirectional Long Short-Term Memory (Bi-LSTM) neural network capable of learning both forward and backward dependencies within historical time-series data. The model processes 30-day multivariate input sequences containing hydrological, meteorological, and geographic features — including rainfall, river discharge, temperature, humidity, water level, and derived features like Rain_7day_sum and Water_vs_Elevation. Based on these sequences, the model predicts the flood risk for the following day as a binary classification task (Flood or No Flood).

The model was trained on 23 years of historical flood data (2001–2024), allowing it to capture long-term seasonal and environmental patterns associated with flood events in the Mahanadi Basin. The project pipeline includes dataset preparation, feature engineering, model training, evaluation through multiple performance metrics (accuracy, precision, recall, F1-score), and visualization using confusion matrix and risk distribution plots.

To make the system more accessible, the trained model has been integrated into an interactive Streamlit web dashboard, allowing users to input district-level queries and instantly receive flood risk predictions along with confidence scores.

The proposed solution offers a practical, scalable, and computationally efficient approach for regional flood forecasting, contributing toward better early warning systems and disaster management preparedness in Odisha.
