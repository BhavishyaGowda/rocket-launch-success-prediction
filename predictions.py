"""
Predictions page component for manual and CSV predictions
"""
import streamlit as st
import pandas as pd
import plotly.express as px
from utils import prepare_prediction_features
from ml_models import make_prediction

def show_manual_prediction(df):
    """Display manual prediction interface"""
    st.subheader("Enter Launch Parameters")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        company = st.selectbox("Company", df['company'].unique() if df is not None else [])
        rocket_type = st.text_input("Rocket Type", "Falcon 9")
        payload_type = st.text_input("Payload Type", "Satellite")
        mission_type = st.text_input("Mission Type", "Communications")
    
    with col2:
        orbit_type = st.selectbox("Orbit Type", df['orbit_type'].unique() if df is not None else [])
        orbit_destination = st.text_input("Orbit Destination", "LEO")
        launch_site = st.text_input("Launch Site", "Kennedy Space Center")
        site_location = st.text_input("Site Location", "Florida, USA")
    
    with col3:
        launch_date = st.date_input("Launch Date")
        launch_time = st.time_input("Launch Time")
        latitude = st.number_input("Latitude", value=28.5, format="%.2f")
        longitude = st.number_input("Longitude", value=-80.5, format="%.2f")
    
    with col4:
        payload_mass = st.number_input("Payload Mass (kg)", value=5000, min_value=0, step=100)
        temperature = st.number_input("Temperature (°C)", value=25, min_value=-50, max_value=50, step=1)
        wind_speed = st.number_input("Wind Speed (km/h)", value=15, min_value=0, max_value=100, step=1)
        engine_count = st.number_input("Engine Count", value=9, min_value=1, max_value=50, step=1)
    
    if st.button("🔮 Predict Success", type="primary", use_container_width=True):
        try:
            # Prepare input data
            row_data = {
                'Company': company,
                'RocketType': rocket_type,
                'PayloadType': payload_type,
                'MissionType': mission_type,
                'OrbitType': orbit_type,
                'OrbitDestination': orbit_destination,
                'LaunchSite': launch_site,
                'SiteLocation': site_location,
                'LaunchDate': str(launch_date),
                'LaunchTime': str(launch_time),
                'Latitude': latitude,
                'Longitude': longitude
            }
            
            # Prepare features
            input_df = prepare_prediction_features(
                row_data, df, 
                st.session_state.label_encoders, 
                st.session_state.feature_names
            )
            
            # Scale the input
            if st.session_state.scaler is not None:
                input_scaled = st.session_state.scaler.transform(input_df)
            else:
                input_scaled = input_df.values
            
            # Make predictions
            predictions, probabilities, ensemble_pred, avg_probability = make_prediction(
                input_scaled, 
                st.session_state.trained_models
            )
            
            if predictions:
                # Display overall prediction
                st.markdown("---")
                st.subheader("🎯 Prediction Results")
                
                overall_result = "Success ✅" if ensemble_pred == 1 else "Failure ❌"
                result_color = "green" if ensemble_pred == 1 else "red"
                
                st.markdown(f"""
                <div style="text-align: center; padding: 30px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; margin-bottom: 20px;">
                    <h1 style="color: white; font-size: 3rem; margin: 0;">Overall Prediction: <span style="color: {result_color};">{overall_result}</span></h1>
                    <h3 style="color: white; margin-top: 10px;">Confidence: {avg_probability:.1f}%</h3>
                    <p style="color: #e0e0e0; margin-top: 10px;">Based on {len(predictions)} trained models</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Display individual model predictions
                st.markdown("### 📊 Individual Model Predictions")
                cols = st.columns(min(3, len(predictions)))
                
                for idx, (model_name, pred) in enumerate(predictions.items()):
                    with cols[idx % min(3, len(predictions))]:
                        result = "Success" if pred == 1 else "Failure"
                        prob = probabilities.get(model_name, 0)
                        
                        # Get model accuracy from training results
                        accuracy = 0
                        if model_name in st.session_state.model_results:
                            accuracy = st.session_state.model_results[model_name]['accuracy'] * 100
                        
                        st.markdown(f"""
                        <div class="metric-card">
                            <h4>{model_name}</h4>
                            <h2 style="color: {'green' if pred == 1 else 'red'};">{result}</h2>
                            <p><strong>Probability:</strong> {prob:.1f}%</p>
                            <p><strong>Model Accuracy:</strong> {accuracy:.2f}%</p>
                        </div>
                        """, unsafe_allow_html=True)
        
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            st.info("💡 Please ensure the model has been trained with the required features.")

def show_csv_prediction(df):
    """Display CSV upload and bulk prediction interface"""
    st.subheader("Upload CSV for Bulk Prediction")
    
    # Provide example CSV template
    st.markdown("#### 📥 Download Example CSV Template")
    example_data = pd.DataFrame([
        {
            'Company': 'SpaceX', 'RocketType': 'Falcon 9',
            'LaunchDate': '2025-06-15', 'LaunchTime': '14:30:00',
            'LaunchSite': 'Kennedy Space Center', 'SiteLocation': 'Florida, USA',
            'OrbitType': 'LEO', 'OrbitDestination': 'ISS',
            'MissionType': 'Resupply', 'PayloadType': 'Cargo',
            'PayloadMass': 15000, 'Latitude': 28.5, 'Longitude': -80.5,
            'WeatherCondition': 'Clear', 'Temperature': 25,
            'WindSpeed': 15, 'PreviousSuccesses': 180, 'PreviousFailures': 2
        }
    ])
    
    csv_template = example_data.to_csv(index=False)
    st.download_button(
        label="📄 Download CSV Template",
        data=csv_template,
        file_name="launch_prediction_template.csv",
        mime="text/csv"
    )
    
    st.markdown("---")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            upload_df = pd.read_csv(uploaded_file)
            st.success(f"✅ Successfully loaded {len(upload_df)} records")
            
            # Show uploaded data info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Launches", len(upload_df))
            with col2:
                st.metric("Unique Companies", upload_df['Company'].nunique() if 'Company' in upload_df.columns else 0)
            with col3:
                st.metric("Features", len(upload_df.columns))
            
            st.markdown("---")
            st.subheader("📋 Uploaded Data Preview")
            st.dataframe(upload_df, use_container_width=True)
            
            if st.button("🔮 Predict All", type="primary", use_container_width=True):
                with st.spinner("Running predictions on all models..."):
                    try:
                        results_df = upload_df.copy()
                        all_predictions = []
                        all_probabilities = []
                        
                        for idx, row in upload_df.iterrows():
                            # Prepare features for this row
                            input_df = prepare_prediction_features(
                                row.to_dict(), df,
                                st.session_state.label_encoders,
                                st.session_state.feature_names
                            )
                            
                            # Scale the input
                            if st.session_state.scaler is not None:
                                input_scaled = st.session_state.scaler.transform(input_df)
                            else:
                                input_scaled = input_df.values
                            
                            # Make predictions
                            row_preds = []
                            row_probs = []
                            
                            for model_name, model in st.session_state.trained_models.items():
                                try:
                                    pred = model.predict(input_scaled)[0]
                                    row_preds.append(pred)
                                    
                                    if hasattr(model, 'predict_proba'):
                                        prob = model.predict_proba(input_scaled)[0][1] * 100
                                        row_probs.append(prob)
                                    else:
                                        row_probs.append(pred * 100 if pred == 1 else 0)
                                except:
                                    row_preds.append(0)
                                    row_probs.append(50)
                            
                            all_predictions.append(row_preds)
                            all_probabilities.append(row_probs)
                        
                        # Add ensemble predictions
                        ensemble_preds = [1 if sum(preds)/len(preds) >= 0.5 else 0 for preds in all_predictions]
                        avg_probs = [sum(probs)/len(probs) for probs in all_probabilities]
                        
                        results_df['Ensemble_Prediction'] = ['Success' if p == 1 else 'Failure' for p in ensemble_preds]
                        results_df['Average_Probability'] = [round(p, 1) for p in avg_probs]
                        
                        st.success("✅ Predictions completed successfully!")
                        
                        # Display prediction statistics
                        st.markdown("---")
                        st.subheader("📊 Prediction Statistics")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            success_count = (results_df['Ensemble_Prediction'] == 'Success').sum()
                            st.metric("Predicted Success", f"{success_count}", 
                                     f"{(success_count/len(results_df)*100):.1f}%")
                        with col2:
                            failure_count = (results_df['Ensemble_Prediction'] == 'Failure').sum()
                            st.metric("Predicted Failure", f"{failure_count}",
                                     f"{(failure_count/len(results_df)*100):.1f}%")
                        with col3:
                            avg_confidence = results_df['Average_Probability'].mean()
                            st.metric("Avg Confidence", f"{avg_confidence:.1f}%")
                        with col4:
                            st.metric("Total Processed", len(results_df))
                        
                        # Visualization
                        st.markdown("---")
                        st.subheader("📈 Prediction Overview")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            pred_counts = results_df['Ensemble_Prediction'].value_counts()
                            fig = px.pie(values=pred_counts.values, names=pred_counts.index,
                                       title='Prediction Distribution',
                                       color_discrete_map={'Success': 'green', 'Failure': 'red'})
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            if 'Company' in results_df.columns:
                                company_preds = results_df.groupby(['Company', 'Ensemble_Prediction']).size().reset_index(name='Count')
                                fig = px.bar(company_preds, x='Company', y='Count', color='Ensemble_Prediction',
                                           title='Predictions by Company',
                                           color_discrete_map={'Success': 'green', 'Failure': 'red'},
                                           barmode='stack')
                                st.plotly_chart(fig, use_container_width=True)
                        
                        # Display results table
                        st.markdown("---")
                        st.subheader("📋 Detailed Prediction Results")
                        st.dataframe(results_df, use_container_width=True, height=400)
                        
                        # Download results
                        st.markdown("---")
                        results_csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Prediction Results (CSV)",
                            data=results_csv,
                            file_name="prediction_results.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    
                    except Exception as e:
                        st.error(f"Error during prediction: {str(e)}")
                        st.info("💡 Please ensure the CSV file has the required columns and the models are trained.")
        
        except Exception as e:
            st.error(f"Error processing file: {e}")
            st.info("💡 Please ensure your CSV file has the required columns. Download the template for reference.")

def show_predictions(df):
    """Main predictions page with tabs"""
    st.title("🔮 Launch Success Prediction")
    st.markdown("Predict rocket launch outcomes using trained ML models")
    
    if st.session_state.trained_models:
        tab1, tab2 = st.tabs(["Manual Input", "CSV Upload"])
        
        with tab1:
            show_manual_prediction(df)
        
        with tab2:
            show_csv_prediction(df)
    else:
        st.info("⚠️ No trained models available. Please train models first in the Model Training page.")
