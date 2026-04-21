"""
Machine Learning models and training functions
"""
import streamlit as st
import time
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import xgboost as xgb

def get_models_dict():
    """Get dictionary of available ML models"""
    return {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000, C=1.0),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5, weights='uniform'),
        'Naive Bayes': GaussianNB(var_smoothing=1e-9),
        'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42, learning_rate_init=0.001),
        'XGBoost': xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
    }

def train_models(X_train, X_test, y_train, y_test, selected_models):
    """Train selected ML models"""
    models_dict = get_models_dict()
    
    results = {}
    trained_models = {}
    
    for model_name in selected_models:
        if model_name in models_dict:
            with st.spinner(f'Training {model_name}...'):
                start_time = time.time()
                model = models_dict[model_name]
                model.fit(X_train, y_train)
                training_time = time.time() - start_time
                
                y_pred = model.predict(X_test)
                
                results[model_name] = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred, zero_division=0),
                    'recall': recall_score(y_test, y_pred, zero_division=0),
                    'f1_score': f1_score(y_test, y_pred, zero_division=0),
                    'confusion_matrix': confusion_matrix(y_test, y_pred),
                    'training_time': training_time
                }
                trained_models[model_name] = model
    
    return results, trained_models

def make_prediction(input_scaled, trained_models):
    """Make predictions using all trained models and return ensemble result"""
    predictions = {}
    probabilities = {}
    
    for model_name, model in trained_models.items():
        try:
            pred = model.predict(input_scaled)[0]
            predictions[model_name] = pred
            
            # Get probability if available
            if hasattr(model, 'predict_proba'):
                prob = model.predict_proba(input_scaled)[0]
                probabilities[model_name] = prob[1] * 100  # Probability of success
            else:
                probabilities[model_name] = (pred * 100) if pred == 1 else ((1-pred) * 100)
        except Exception as e:
            st.warning(f"Could not get prediction from {model_name}: {str(e)}")
    
    # Calculate ensemble prediction (majority vote)
    if predictions:
        ensemble_pred = 1 if sum(predictions.values()) / len(predictions) >= 0.5 else 0
        avg_probability = sum(probabilities.values()) / len(probabilities)
        return predictions, probabilities, ensemble_pred, avg_probability
    
    return {}, {}, 0, 0

def get_model_options():
    """Get model options with descriptions"""
    return {
        'Logistic Regression': 'Linear Model',
        'K-Nearest Neighbors': 'Instance Model',
        'Naive Bayes': 'Probabilistic Model',
        'Neural Network': 'Deep Model',
        'XGBoost': 'Ensemble Model'
    }
