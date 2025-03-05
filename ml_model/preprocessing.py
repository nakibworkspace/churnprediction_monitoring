import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub
import os
import shutil

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score,
    classification_report, confusion_matrix
)

path = kagglehub.dataset_download('blastchar/telco-customer-churn', force_download=True)
destination_path = '/root/code/Dataset'
shutil.copytree(path, destination_path, dirs_exist_ok=True)

def load_and_preprocess_data():
    df= pd.read_csv("/root/code/Dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df.head()

    df.shape
    df.info()
    df.columns
    df.describe()
    df.nunique()

    df = df.drop('customerID', axis=1)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors="coerce")

    print("Missing Values Count:")
    print(df.isnull().sum())

    print("Missing Values Count:")
    print(df.isnull().sum())

    df.drop(df[df['TotalCharges'].isnull()].index, inplace=True)
    df.reset_index(drop=True, inplace=True)

    df.replace('No internet service', 'No', inplace=True)
    df.replace('No phone service', 'No', inplace=True)

    # Display unique values in categorical columns
    for i in df.columns:
        if df[i].dtypes=="object":
            print(f'{i} : {df[i].unique()}')
            print("****************************************************")

    # Convert gender to numeric
    df['gender'].replace({'Female':1,'Male':0}, inplace=True)

    # One-hot encoding for multi-category variables
    # Handle variables with more than 2 categories
    more_than_2 = ['InternetService' ,'Contract' ,'PaymentMethod']
    df = pd.get_dummies(data=df, columns=more_than_2)

    # Feature scaling for numerical columns
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()

    # Scale continuous variables
    large_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    df[large_cols] = scaler.fit_transform(df[large_cols])

    # Convert binary categories to numeric
    two_cate = ['Partner', 'Dependents', 'PhoneService', 'MultipleLines', 
                'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                'TechSupport', 'StreamingTV', 'StreamingMovies', 
                'PaperlessBilling', 'Churn']
    for i in two_cate:
        df[i].replace({"No":0, "Yes":1}, inplace=True)

    return df    