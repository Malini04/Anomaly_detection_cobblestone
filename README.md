# Efficient Data Stream Anomaly Detection

This project implements anomaly detection techniques on a continuous data stream, simulating real-time sequences of floating-point numbers. The focus is on identifying unusual patterns, such as exceptionally high values or deviations from the norm.

## Project Title
Efficient Data Stream Anomaly Detection

## Project Description
The goal is to develop a Python script capable of detecting anomalies in a data stream. This stream could represent various metrics such as financial transactions or system metrics, and will include regular patterns, seasonal elements, and random noise.

For my project,i have used Fraud financial Detection: 
Anomaly Detection in Credit Card Transactions

### Project Overview
The project aims to detect anomalies in credit card transactions, focusing on:
- Identifying fraudulent transactions using **Isolation Forest** and **Local Outlier Factor (LOF)**.
- Comparing the effectiveness of both algorithms in detecting anomalies.
- Providing statistics on the detected fraudulent transactions.

The project also includes robust error handling and documentation to ensure ease of understanding and reproducibility.

### Dataset
The dataset used for this project is the **Credit Card Fraud Detection dataset** available on Kaggle. It contains transactions with various features and a label indicating whether each transaction is fraudulent or legitimate.

### Sample of the Data
```bash
  Time, V1, V2, V3, ..., V28, Amount, Class
  0, -1.35980713, 1.19185711, ..., -0.05395037, 149.62, 0
  1, -1.35835305, 2.29212882, ..., 0.25542586, 2.69, 0
  2, -0.96662024, 0.36188434, ..., 0.19811233, 378.66, 0
  ...
```

## Getting Started

### Prerequisites
Make sure you have the following installed:

- Python 3.x
- pip (Python package manager)

### Project Structure
```bash
Anomaly_detection_cobblestone/
│
├── anomaly_detection.py        # The main script for anomaly detection.
├── requirements.txt            # A list of required Python packages.
├── .gitignore                  # A file that specifies which files and directories to ignore in the repository.
├── README.md                   # This file.
└── creditcard.csv              # The dataset for credit card transactions.
```

### Download the Dataset

1. Go to the Kaggle dataset page: [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
2. Sign in to your Kaggle account (or create one if you don't have an account).
3. Click on the "Download" button to download the dataset as a ZIP file.
4. Extract the ZIP file, and place the `creditcard.csv` file in the project directory.

### Clone the Repository
To clone the repository, use the following command:
```bash
git clone https://github.com/Malini04/Anomaly_detection_cobblestone.git
```

### Install Requirements
Navigate to the project directory and install the required packages using the following command:
```bash
cd Anomaly_detection_cobblestone
```
### Install Dependencies
```bash
pip install -r requirements.txt
```

### Running the Project
You can run the anomaly detection script using the command line:
```bash
python anomaly_detection.py
```

### Results and Statistics
After running the script, you will see the output showing the number of anomalies detected by each algorithm. The following statistics will be printed:
```bash
Anomaly Detection Statistics:
Isolation Forest - Detected Anomalies: X
Local Outlier Factor - Detected Anomalies: Y
```

### References

- Credit Card Fraud Detection Dataset: Available on Kaggle
- Isolation Forest: Documentation for Isolation Forest in scikit-learn - scikit-learn Isolation Forest
- Local Outlier Factor: Documentation for LOF in scikit-learn - scikit-learn LOF
- Matplotlib: Official documentation for the Python plotting library - Matplotlib
- Pandas: Data analysis library for Python - Pandas Documentation
- Numpy: Fundamental package for numerical computing in Python - NumPy Documentation

