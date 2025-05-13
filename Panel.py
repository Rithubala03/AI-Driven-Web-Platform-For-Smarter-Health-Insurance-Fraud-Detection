import panel as pn
import pandas as pd
import pickle
import sqlite3
import hashlib
from sklearn.naive_bayes import GaussianNB
import io
import numpy as np

pn.extension()

# Database Setup
conn = sqlite3.connect("fraud_detection.db", check_same_thread=False)
cursor = conn.cursor()

# Drop tables if they exist (only do this during development)
#cursor.execute("DROP TABLE IF EXISTS users")
#cursor.execute("DROP TABLE IF EXISTS customer_data")
#cursor.execute("DROP TABLE IF EXISTS predictions")

# Create tables with correct schema
cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE,
    password TEXT
)""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS customer_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    customer_id TEXT,
    date TEXT,
    age INTEGER,
    name TEXT,
    diagnosis TEXT,
    hospital_type TEXT,
    previous_claims INTEGER,
    claim_amount REAL,
    FOREIGN KEY(user_id) REFERENCES users(id)
)""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    customer_id TEXT,
    name TEXT,
    claim_amount REAL,
    age INTEGER,
    diagnosis TEXT,
    hospital_type TEXT,
    previous_claims INTEGER,
    prediction TEXT,
    probability REAL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(user_id) REFERENCES users(id)
)""")

conn.commit()


# Load model and encoders
try:
    with open("naive_bayes.pkl", "rb") as f:
        model = pickle.load(f)
    with open("label_encoders.pkl", "rb") as f:
        label_encoders = pickle.load(f)
except FileNotFoundError as e:
    print(f"Error loading model files: {e}")
    exit(1)

# Helper functions
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def authenticate(username, password):
    cursor.execute("SELECT id, password FROM users WHERE username=?", (username,))
    user = cursor.fetchone()
    return (user[0], user[1] == hash_password(password)) if user else (None, False)

def preprocess_input(claim_amount, age, diagnosis, hospital_type, previous_claims):
    try:
        diagnosis_encoded = label_encoders["Diagnosis"].transform([diagnosis])[0] 
    except:
        diagnosis_encoded = 0
        
    try:
        hospital_type_encoded = label_encoders["HospitalType"].transform([hospital_type])[0]
    except:
        hospital_type_encoded = 0
        
    return pd.DataFrame([[claim_amount, age, diagnosis_encoded, hospital_type_encoded, previous_claims]],
                       columns=["ClaimAmount", "Age", "Diagnosis", "HospitalType", "PreviousClaims"])

def predict_fraud(user_id, customer_id, name, diagnosis, hospital_type, claim_amount):
    # Get customer's basic profile (age and previous claims)
    cursor.execute("""
        SELECT age, previous_claims 
        FROM customer_data 
        WHERE user_id=? AND customer_id=? AND name=?
        ORDER BY date DESC LIMIT 1
    """, (user_id, customer_id, name))
    
    profile = cursor.fetchone()
    if not profile:
        return "Error: Customer profile not found", 0.0

    age, previous_claims = profile
    
    # Preprocess data for model
    input_data = preprocess_input(
        claim_amount=float(claim_amount),
        age=int(age),
        diagnosis=diagnosis,
        hospital_type=hospital_type,
        previous_claims=int(previous_claims))
    
    # Get base prediction
    try:
        prediction_prob = model.predict_proba(input_data.to_numpy())[0][1]
    except:
        prediction_prob = 0.0
    
    # Apply business rules
    rules = {
        'high_amount': {'threshold': 300000, 'adjustment': 0.10},
        'frequent_claims': {'threshold': 5, 'adjustment': 0.15},
        'suspicious_diagnosis': {'list': ['cancer', 'heart disease'], 'adjustment': 0.20},
        'private_hospital': {'adjustment': 0.10}
    }
    
    # High amount rule
    if float(claim_amount) > rules['high_amount']['threshold']:
        prediction_prob += rules['high_amount']['adjustment']
    
    # Frequent claims rule
    if int(previous_claims) > rules['frequent_claims']['threshold']:
        prediction_prob += rules['frequent_claims']['adjustment']
    
    # Suspicious diagnosis rule
    if diagnosis.lower() in rules['suspicious_diagnosis']['list']:
        prediction_prob += rules['suspicious_diagnosis']['adjustment']
    
    # Private hospital rule
    if hospital_type.lower() == 'private':
        prediction_prob += rules['private_hospital']['adjustment']
    
    # Cap probability between 0-1
    prediction_prob = max(0, min(1, prediction_prob))
    
    # Determine result
    result = "Fraud" if prediction_prob > 0.35 else "Legitimate"
    
    # Store prediction
    cursor.execute("""
    INSERT INTO predictions (
        user_id, customer_id, name, claim_amount, age, diagnosis,
        hospital_type, previous_claims, prediction, probability
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        user_id, customer_id, name, claim_amount, age, diagnosis,
        hospital_type, previous_claims, result, prediction_prob
    ))
    conn.commit()
    
    return result, prediction_prob

# UI Components
login_username = pn.widgets.TextInput(name="Username", placeholder="Enter username")
login_password = pn.widgets.PasswordInput(name="Password", placeholder="Enter password")
login_button = pn.widgets.Button(name="Login", button_type="primary")
login_message = pn.pane.Markdown()

signup_username = pn.widgets.TextInput(name="New Username", placeholder="Choose username")
signup_password = pn.widgets.PasswordInput(name="New Password", placeholder="Create password")
signup_button = pn.widgets.Button(name="Signup", button_type="success")
signup_message = pn.pane.Markdown()

# Prediction Form Widgets
customer_id = pn.widgets.TextInput(name="Customer ID", placeholder="Enter customer ID")
name = pn.widgets.TextInput(name="Customer Name", placeholder="Enter customer name")
diagnosis = pn.widgets.TextInput(name="Diagnosis", placeholder="Enter medical diagnosis")
hospital_type = pn.widgets.Select(name="Hospital Type", 
                                options=['Government', 'Private', 'Charity', 'Clinic', 'Other'],
                                value='Government')
claim_amount = pn.widgets.TextInput(name="Claim Amount", placeholder="Enter claim amount")
submit_button = pn.widgets.Button(name="Predict", button_type="primary")
output = pn.pane.Markdown()

# File Upload Widgets
file_input = pn.widgets.FileInput(accept=".xlsx,.csv", name="Upload Customer Data")
upload_button = pn.widgets.Button(name="Upload Data", button_type="success")
upload_output = pn.pane.Markdown()

# Navigation
login_link = pn.widgets.Button(name="Don't have an account? Sign up", button_type="light")
signup_link = pn.widgets.Button(name="Already have an account? Login", button_type="light")

# Functions
def process_excel(event):
    if not file_input.value:
        upload_output.object = "### Please select a file first"
        upload_output.style = {'color': 'red'}
        return

    try:
        user_id, _ = authenticate(login_username.value, login_password.value)
        if not user_id:
            upload_output.object = "### Error: Authentication failed"
            upload_output.style = {'color': 'red'}
            return

        df = pd.read_excel(io.BytesIO(file_input.value))
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        
        required_cols = {'customer_id', 'date', 'age', 'name', 'diagnosis', 'hospital_type', 'previous_claims', 'claim_amount'}
        if not required_cols.issubset(set(df.columns)):
            missing = required_cols - set(df.columns)
            upload_output.object = f"### Error: Missing columns: {', '.join(missing)}"
            upload_output.style = {'color': 'red'}
            return

        for _, row in df.iterrows():
            cursor.execute("""
            INSERT INTO customer_data (
                user_id, customer_id, date, age, name, diagnosis, 
                hospital_type, previous_claims, claim_amount
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                user_id, 
                str(row['customer_id']).strip(),
                str(row['date']),
                int(row['age']),
                str(row['name']).strip(),
                str(row['diagnosis']).strip(),
                str(row['hospital_type']).strip(),
                int(row['previous_claims']),
                float(row['claim_amount'])
            ))
        
        conn.commit()
        upload_output.object = "### Data uploaded successfully! ✅"
        upload_output.style = {'color': 'green'}
        
    except Exception as e:
        upload_output.object = f"### Error: {str(e)}"
        upload_output.style = {'color': 'red'}

def on_submit(event):
    try:
        user_id, authenticated = authenticate(login_username.value, login_password.value)
        if not authenticated:
            output.object = "### Error: Please login first"
            output.style = {'color': 'red'}
            return

        # Validate all required fields
        required_fields = {
            'Customer ID': customer_id.value.strip(),
            'Name': name.value.strip(),
            'Diagnosis': diagnosis.value.strip(),
            'Hospital Type': hospital_type.value,
            'Claim Amount': claim_amount.value.strip()
        }
        
        if not all(required_fields.values()):
            missing = [k for k, v in required_fields.items() if not v]
            output.object = f"### Error: Missing fields: {', '.join(missing)}"
            output.style = {'color': 'red'}
            return

        try:
            claim_amt = float(claim_amount.value)
        except ValueError:
            output.object = "### Error: Claim amount must be a number"
            output.style = {'color': 'red'}
            return

        result, probability = predict_fraud(
            user_id=user_id,
            customer_id=customer_id.value.strip(),
            name=name.value.strip(),
            diagnosis=diagnosis.value.strip(),
            hospital_type=hospital_type.value,
            claim_amount=claim_amt
        )

        # Display results
        if "Error:" in result:
            output.object = f"### {result}"
            output.style = {'color': 'red'}
        else:
            output.object = f"""
            ### Prediction Results:
            - **Status**: {result}
            - **Probability**: {probability:.1%}
            """
            output.style = {'color': 'green' if result == "Legitimate" else 'red'}
            
    except Exception as e:
        output.object = f"### Error: {str(e)}"
        output.style = {'color': 'red'}

def login_user(event):
    username = login_username.value.strip()
    password = login_password.value.strip()
    
    if not username or not password:
        login_message.object = "### Please enter both username and password"
        login_message.style = {'color': 'red'}
        return
        
    user_id, authenticated = authenticate(username, password)
    if authenticated:
        login_message.object = "### Login successful! ✅"
        login_message.style = {'color': 'green'}
        main_area[:] = [dashboard]
    else:
        login_message.object = "### Error: Invalid credentials"
        login_message.style = {'color': 'red'}

def signup_user(event):
    username = signup_username.value.strip()
    password = signup_password.value.strip()
    
    if not username or not password:
        signup_message.object = "### Please enter both username and password"
        signup_message.style = {'color': 'red'}
        return
        
    if len(password) < 8:
        signup_message.object = "### Password must be at least 8 characters"
        signup_message.style = {'color': 'red'}
        return
        
    try:
        cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", 
                      (username, hash_password(password)))
        conn.commit()
        signup_message.object = "### Signup successful! Please login. ✅"
        signup_message.style = {'color': 'green'}
    except sqlite3.IntegrityError:
        signup_message.object = "### Error: Username already exists"
        signup_message.style = {'color': 'red'}

# Navigation functions
def go_to_signup(event): main_area[:] = [signup_page]
def go_to_login(event): main_area[:] = [login_page]

# Connect buttons
login_button.on_click(login_user)
signup_button.on_click(signup_user)
upload_button.on_click(process_excel)
submit_button.on_click(on_submit)
login_link.on_click(go_to_signup)
signup_link.on_click(go_to_login)

# Pages
login_page = pn.Column(
    "# Login",
    login_username,
    login_password,
    login_button,
    login_message,
    login_link
)

signup_page = pn.Column(
    "# Signup",
    signup_username,
    signup_password,
    signup_button,
    signup_message,
    signup_link
)

dashboard = pn.Column(
    "# Medical Insurance Fraud Detection",
    "## Upload Customer History",
    pn.Row(file_input, upload_button),
    upload_output,
    "## Predict Fraud Risk",
    customer_id,
    name,
    diagnosis,
    hospital_type,
    claim_amount,
    submit_button,
    output
)

# Main app
main_area = pn.Column(login_page)

app = pn.template.FastListTemplate(
    title="Fraud Detection System",
    main=[main_area],
    header_background="#1f77b4",
    accent_base_color="#1f77b4"
)

def cleanup():
    conn.close()
    print("Database connection closed")

import atexit
atexit.register(cleanup)

app.servable()
