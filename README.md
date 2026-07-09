# 🌊 NeerDrishti – Flood Forecasting System

A deep learning-based flood forecasting system for the Mahanadi River Basin, Odisha, India.

## 📌 Project Overview

NeerDrishti is a deep learning-based flood risk prediction system designed to forecast flood events using historical hydrological, meteorological, and geographic data from the Mahanadi River Basin.

The project addresses the challenge of predicting floods by learning complex temporal patterns from historical data using a Bidirectional Long Short-Term Memory (Bi-LSTM) neural network.

## 🧠 Model Approach

The model uses a Bi-LSTM architecture to learn forward and backward dependencies from multivariate time-series data.

- Input: 30-day historical sequences
- Output: Next-day flood risk prediction (Flood / No Flood)
- Dataset: 23 years of historical flood data (2001–2024)
- Region: Mahanadi River Basin, Odisha

## 📊 Features Used

The model uses hydrological, meteorological, and geographic features such as:

- Rainfall
- River discharge
- Temperature
- Humidity
- Water level
- Elevation
- District information
- Derived features:
  - Rain_7day_sum
  - Water_vs_Elevation

## 🚀 Features

- Flood risk prediction using Bi-LSTM
- Historical data-based forecasting
- Confidence score generation
- Interactive Streamlit dashboard
- District-level flood risk visualization

## 🛠 Technologies Used

- Python
- TensorFlow
- Keras
- Scikit-learn
- Pandas
- NumPy
- Streamlit

## 📂 Repository Structure

NeerDrishti/
├── README.md                     # Project documentation
├── requirements.txt              # Project dependencies
├── main.ipynb                    # Model training and evaluation
├── neerdrishti_app.py            # Streamlit web application
├── mahanadi_bilstm_model.h5      # Trained Bi-LSTM model
├── scaler.pkl                    # Saved feature scaler
└── label_encoders.pkl            # Saved label encoders


## ⚙️ Installation

1. Clone the repository:

```bash
git clone <repository-url>

2. Navigate to the project folder:

   cd NeerDrishti

3. Install dependencies:

   pip install -r requirements.txt

4. Run the application:

   streamlit run neerdrishti_app.py


## 📈 Evaluation

The model performance was evaluated using:

• Accuracy
• Precision
• Recall
• F1-score
• Confusion Matrix


## 🔮 Future Enhancements

- Real-time weather and river data integration
- GIS-based flood visualization
- Cloud deployment
- Enhanced prediction accuracy using additional datasets