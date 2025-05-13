import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Function to load and preprocess data
def preprocess_data(file_path):
    # Load dataset
    df = pd.read_csv(file_path)

    # Select relevant features
    selected_columns = [
        "ClaimAmount",
        "Age",
        "Diagnosis",
        "HospitalType",
        "PreviousClaims",
        "FraudFlag",
    ]
    df = df[selected_columns]

    # Handle categorical data using Label Encoding
    label_encoders = {}
    categorical_columns = ["Diagnosis", "HospitalType"]
    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Handle missing values by filling with mean for numerical columns
    df = df.fillna(df.mean())

    return df, label_encoders

# Function to split data into train and test
def split_data(df):
    X = df.iloc[:, :-1].values  
    y = df.iloc[:, -1].values  

    # Split the data into train and test sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    return X_train, X_test, y_train, y_test

# Function to train Naïve Bayes model
def train_naive_bayes(X_train, y_train):
    model = GaussianNB()
    model.fit(X_train, y_train)

    # Save the trained model to a pickle file
    with open("naive_bayes.pkl", "wb") as f:
        pickle.dump(model, f)

    return model

# Function to evaluate the model and check overfitting/underfitting
def evaluate_model(model, X_train, X_test, y_train, y_test):
    # Get train and test accuracy
    train_accuracy = accuracy_score(y_train, model.predict(X_train))
    test_accuracy = accuracy_score(y_test, model.predict(X_test))

    print(f"Train Accuracy: {train_accuracy:.2f}")
    print(f"Test Accuracy: {test_accuracy:.2f}")

    # Check model performance
    if train_accuracy > 0.9 and test_accuracy < 0.7:
        print("The model is overfitting.")
    elif train_accuracy < 0.7 and test_accuracy < 0.7:
        print("The model is underfitting.")
    else:
        print("The model has a good fit.")

# Function to preprocess data, split it, train, evaluate, and save the model
def main():
    # Load and preprocess the data
    df, label_encoders = preprocess_data("synthetic_medical_aid_claims.csv")

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = split_data(df)

    # Train the Naïve Bayes model
    model = train_naive_bayes(X_train, y_train)

    # Evaluate the model
    evaluate_model(model, X_train, X_test, y_train, y_test)

    # Save label encoders for future use
    with open("label_encoders.pkl", "wb") as f:
        pickle.dump(label_encoders, f)

if __name__ == "__main__":
    main()
