"""
Utility functions for data processing and exports
"""
import streamlit as st
import pandas as pd
import numpy as np
import io
import base64
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

@st.cache_data
def load_data():
    """Load the rocket launch dataset"""
    url = "https://hebbkx1anhila5yf.public.blob.vercel-storage.com/launches%20%281%29-V1kwKSujKOshGTAcnqkYx3m6D3QKbv.csv"
    try:
        df = pd.read_csv(url)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def preprocess_data(df, test_size=0.2, random_state=42):
    """Preprocess the data for machine learning"""
    # Handle missing values
    df = df.dropna(subset=['success'])
    
    # Convert datetime and extract features
    df['launch_datetime'] = pd.to_datetime(df['launch_datetime'], errors='coerce')
    df['launch_year'] = df['launch_datetime'].dt.year
    df['launch_month'] = df['launch_datetime'].dt.month
    df['launch_day_of_year'] = df['launch_datetime'].dt.dayofyear
    df['launch_hour'] = df['launch_datetime'].dt.hour
    
    # Encode categorical variables
    categorical_columns = ['company', 'rocket_type', 'payload_type', 'mission_type', 
                          'orbit_type', 'orbit_destination', 'launch_site', 'site_location']
    
    label_encoders = {}
    for col in categorical_columns:
        if col in df.columns:
            le = LabelEncoder()
            df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
    
    # Create historical success rate features
    company_success_rate = df.groupby('company')['success'].mean()
    df['company_historical_success_rate'] = df['company'].map(company_success_rate)
    
    orbit_success_rate = df.groupby('orbit_type')['success'].mean()
    df['orbit_historical_success_rate'] = df['orbit_type'].map(orbit_success_rate)
    
    # Select features
    feature_columns = [
        'launch_year', 'launch_month', 'launch_day_of_year',
        'company_encoded', 'rocket_type_encoded', 'payload_type_encoded',
        'mission_type_encoded', 'orbit_type_encoded', 'orbit_destination_encoded',
        'launch_site_encoded', 'site_location_encoded',
        'company_historical_success_rate', 'orbit_historical_success_rate',
        'lat', 'lon'
    ]
    
    available_features = [col for col in feature_columns if col in df.columns]
    X = df[available_features].fillna(0)
    y = df['success']
    
    # Split and scale
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, label_encoders, available_features, df

def export_to_csv(df):
    """Export dataframe to CSV"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="rocket_launch_data.csv">Download CSV</a>'

def export_to_excel(df):
    """Export dataframe to Excel"""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Launch Data')
    output.seek(0)
    b64 = base64.b64encode(output.read()).decode()
    return f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="rocket_launch_data.xlsx">Download Excel</a>'

def prepare_prediction_features(row_data, df, label_encoders, feature_names):
    """Prepare features for prediction from input data"""
    # Extract datetime features
    try:
        launch_datetime = pd.to_datetime(f"{row_data.get('LaunchDate', '2025-01-01')} {row_data.get('LaunchTime', '12:00:00')}")
        launch_year = launch_datetime.year
        launch_month = launch_datetime.month
        launch_day_of_year = launch_datetime.dayofyear
    except:
        launch_year = 2025
        launch_month = 1
        launch_day_of_year = 1
    
    # Create feature dictionary
    input_data = {
        'launch_year': launch_year,
        'launch_month': launch_month,
        'launch_day_of_year': launch_day_of_year,
        'lat': row_data.get('Latitude', 0),
        'lon': row_data.get('Longitude', 0)
    }
    
    # Encode categorical features
    categorical_mapping = {
        'company': row_data.get('Company', 'Unknown'),
        'rocket_type': row_data.get('RocketType', 'Unknown'),
        'payload_type': row_data.get('PayloadType', 'Unknown'),
        'mission_type': row_data.get('MissionType', 'Unknown'),
        'orbit_type': row_data.get('OrbitType', 'Unknown'),
        'orbit_destination': row_data.get('OrbitDestination', 'Unknown'),
        'launch_site': row_data.get('LaunchSite', 'Unknown'),
        'site_location': row_data.get('SiteLocation', 'Unknown')
    }
    
    for col, value in categorical_mapping.items():
        if col in label_encoders:
            le = label_encoders[col]
            try:
                encoded_value = le.transform([str(value)])[0]
            except ValueError:
                encoded_value = 0
            input_data[f'{col}_encoded'] = encoded_value
    
    # Add historical success rates
    company = row_data.get('Company', 'Unknown')
    orbit = row_data.get('OrbitType', 'Unknown')
    
    if df is not None:
        company_success_rate = df[df['company'] == company]['success'].mean() if company in df['company'].values else 0.85
        orbit_success_rate = df[df['orbit_type'] == orbit]['success'].mean() if orbit in df['orbit_type'].values else 0.85
    else:
        company_success_rate = 0.85
        orbit_success_rate = 0.85
    
    input_data['company_historical_success_rate'] = company_success_rate
    input_data['orbit_historical_success_rate'] = orbit_success_rate
    
    # Create DataFrame with correct feature order
    input_df = pd.DataFrame([input_data])
    
    # Ensure all expected features are present
    for feature in feature_names:
        if feature not in input_df.columns:
            input_df[feature] = 0
    
    # Select only the features used in training
    input_df = input_df[feature_names]
    
    return input_df
