# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from pylab import rcParams

# Set plot parameters for visualization
rcParams['figure.figsize'] = 14, 8
RANDOM_SEED = 42  # Seed for reproducibility
LABELS = ["Normal", "Fraud"]  # Labels for class distribution

# Load the dataset
# change the name of your dataset accordingly...
df = pd.read_csv('creditcard.csv')
print(df.head())  # Preview the first few rows of the dataset

# Quick overview of the dataset structure and types
print(df.info())

# Check for missing values in the dataset
print(df.isnull().sum())

# Class distribution (Normal vs Fraud)
count_classes = df['Class'].value_counts(sort=True)

# Plot the class distribution
count_classes.plot(kind='bar', rot=0)
plt.title("Transaction Class Distribution")  # Title for the plot
plt.xticks(range(2), LABELS)  # Set x-tick labels
plt.xlabel("Class")  # Label for x-axis
plt.ylabel("Frequency")  # Label for y-axis
plt.show()  # Display the plot

# Splitting dataset into Fraud and Normal transactions
fraud = df[df['Class'] == 1]  # Filter for fraud transactions
normal = df[df['Class'] == 0]  # Filter for normal transactions
print(f"Fraud transactions: {fraud.shape}, Normal transactions: {normal.shape}")

# Take a sample (10%) of the dataset for faster processing
df_sample = df.sample(frac=0.1, random_state=RANDOM_SEED)
print(f"Sample shape: {df_sample.shape}, Original shape: {df.shape}")

# Determine fraud/valid transaction counts in the sample
Fraud = df_sample[df_sample['Class'] == 1]  # Filter fraud from the sample
Valid = df_sample[df_sample['Class'] == 0]  # Filter valid transactions
outlier_fraction = len(Fraud) / float(len(Valid))  # Calculate outlier fraction
print(f"Outlier fraction: {outlier_fraction}")

# Create independent and dependent features
columns = df_sample.columns.tolist()  # Get all column names
columns = [col for col in columns if col not in ["Class"]]  # Exclude target variable
target = "Class"  # Define the target variable

# Define a random state for reproducibility
state = np.random.RandomState(RANDOM_SEED)

# Split the sample into features (X) and target variable (Y)
X = df_sample[columns]
Y = df_sample[target]

# Print the shape of features and target variable
print(f"X shape: {X.shape}, Y shape: {Y.shape}")

# Define the anomaly detection models without SVM
classifiers = {
    "Isolation Forest": IsolationForest(n_estimators=100, max_samples=len(X),
                                        contamination=outlier_fraction, random_state=RANDOM_SEED),
    "Local Outlier Factor": LocalOutlierFactor(n_neighbors=20, algorithm='auto', 
                                               leaf_size=30, metric='minkowski',
                                               p=2, metric_params=None, contamination=outlier_fraction)
}

# Iterate over each classifier
for clf_name, clf in classifiers.items():
    # Fit the data and tag outliers
    if clf_name == "Local Outlier Factor":
        y_pred = clf.fit_predict(X)  # Fit LOF model
        # Adjust predictions: LOF assigns -1 for outliers
        y_pred = np.where(y_pred == -1, 1, 0)  # Convert -1 to 1 (fraud)

    else:  # Isolation Forest
        clf.fit(X)  # Fit Isolation Forest model
        y_pred = clf.predict(X)  # Get predictions
        # Adjust predictions: Isolation Forest assigns -1 for outliers
        y_pred = np.where(y_pred == -1, 1, 0)  # Convert -1 to 1 (fraud)

    # Count misclassified samples
    n_errors = (y_pred != Y).sum()

    # Output results
    print(f"{clf_name}: Number of errors: {n_errors}")
    print(f"Accuracy Score: {accuracy_score(Y, y_pred):.4f}")  # Calculate and print accuracy
    print("Classification Report:")  # Print classification metrics
    print(classification_report(Y, y_pred))  # Detailed classification report
