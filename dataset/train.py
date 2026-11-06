import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import os
import joblib
# Read csv file
data = Path(r"D:\TNS PROJECT\dataset\credit_card_frauds.csv")
df=pd.read_csv(data)

# performing EDA

print(df.shape) 

print(df.info())

print(df.describe())

print(df.isnull().sum())

print(df.duplicated().sum())


# Data Preprocessing

# Convert transaction date into datetime
df["trans_date_trans_time"] = pd.to_datetime(
    df["trans_date_trans_time"]
)

# Extract time features
df["trans_hour"] = df["trans_date_trans_time"].dt.hour
df["trans_day"] = df["trans_date_trans_time"].dt.day
df["trans_month"] = df["trans_date_trans_time"].dt.month

# Remove unnecessary text columns
df = df.drop(
    [
        "trans_date_trans_time",
        "merchant",
        "city",
        "state",
        "job",
        "dob",
        "trans_num"
    ],
    axis=1
)

# Convert category into numerical values
df = pd.get_dummies(
    df,
    columns=["category"],
    dtype=int
)


#  Split and Train, Test
x=df.drop("is_fraud",axis=1) 
y=df["is_fraud"] 

# Train anf testing

x_train,x_test,y_train,y_test = train_test_split(
    x,y,
    test_size=0.2,
    random_state=42,
    stratify = y
)

# Feature Scaling

scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)

x_test = scaler.transform(x_test)

# Model Training
model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

model.fit(x_train, y_train)

# Training prediction
x_train_pred = model.predict(x_train)

training_acc = accuracy_score(y_train, x_train_pred)

print("Training Accuracy:", training_acc)

# Testing prediction
x_test_pred = model.predict(x_test)

testing_acc = accuracy_score(y_test, x_test_pred)

print("Testing Accuracy:", testing_acc)
print("Precision:", precision_score(y_test, x_test_pred))
print("Recall:", recall_score(y_test, x_test_pred))
print("F1 Score:", f1_score(y_test, x_test_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, x_test_pred))

# Save trained model and preprocessing objects

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

model_data = {
    "model": model,
    "scaler": scaler,
    "feature_names": x.columns.tolist()
}

joblib.dump(
    model_data,
    os.path.join(MODEL_DIR, "fraud_detection_model.pkl")
)

print("\nModel and preprocessing objects saved successfully!")