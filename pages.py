"""
Page components for the Rocket Launch Prediction Dashboard
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils import export_to_csv, export_to_excel, prepare_prediction_features, preprocess_data
from ml_models import train_models, make_prediction, get_model_options

def show_dashboard(df):
    """Display the main dashboard page"""
    st.title("🚀 Launch Dashboard")
    
    if df is not None:
        # Calculate KPIs
        total_launches = len(df)
        success_rate = (df['success'].sum() / total_launches * 100) if total_launches > 0 else 0
        total_companies = df['company'].nunique()
        total_orbit_types = df['orbit_type'].nunique()
        
        # Display KPIs
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Launches", f"{total_launches:,}")
        with col2:
            st.metric("Success Rate", f"{success_rate:.2f}%")
        with col3:
            st.metric("Total Companies", total_companies)
        with col4:
            st.metric("Total Orbit Types", total_orbit_types)
        
        # Export options
        st.markdown("---")
        st.subheader("📥 Export Options")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📄 Export to CSV"):
                st.markdown(export_to_csv(df), unsafe_allow_html=True)
        with col2:
            if st.button("📊 Export to Excel"):
                st.markdown(export_to_excel(df), unsafe_allow_html=True)
        
        # Tabs for different visualizations
        st.markdown("---")
        tab1, tab2, tab3 = st.tabs(["Success by Company", "Success by Orbit", "Yearly Trends"])
        
        with tab1:
            st.subheader("Success Rate by Company")
            company_stats = df.groupby('company').agg({
                'success': ['count', 'sum', 'mean']
            }).reset_index()
            company_stats.columns = ['Company', 'Total Launches', 'Successful', 'Success Rate']
            company_stats['Success Rate'] = (company_stats['Success Rate'] * 100).round(2)
            company_stats = company_stats.sort_values('Total Launches', ascending=False).head(10)
            
            fig = px.bar(company_stats, x='Company', y='Success Rate', 
                        title='Top 10 Companies by Success Rate',
                        color='Success Rate',
                        color_continuous_scale='RdYlGn',
                        text='Success Rate')
            fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            st.dataframe(company_stats, use_container_width=True, hide_index=True)
        
        with tab2:
            st.subheader("Success by Orbit Type")
            orbit_stats = df.groupby('orbit_type').agg({
                'success': ['count', 'mean']
            }).reset_index()
            orbit_stats.columns = ['Orbit Type', 'Total', 'Success Rate']
            orbit_stats['Success Rate'] = (orbit_stats['Success Rate'] * 100).round(2)
            orbit_stats = orbit_stats.sort_values('Total', ascending=False).head(10)
            
            fig = px.pie(orbit_stats, values='Total', names='Orbit Type',
                        title='Launch Distribution by Orbit Type',
                        hover_data=['Success Rate'])
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            st.dataframe(orbit_stats, use_container_width=True, hide_index=True)
        
        with tab3:
            st.subheader("Yearly Trends")
            df['year'] = pd.to_datetime(df['launch_datetime']).dt.year
            yearly_stats = df.groupby('year').agg({
                'success': ['count', 'mean']
            }).reset_index()
            yearly_stats.columns = ['Year', 'Total Launches', 'Success Rate']
            yearly_stats['Success Rate'] = (yearly_stats['Success Rate'] * 100).round(2)
            
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(
                go.Bar(x=yearly_stats['Year'], y=yearly_stats['Total Launches'], 
                      name='Total Launches', marker_color='lightblue'),
                secondary_y=False
            )
            fig.add_trace(
                go.Scatter(x=yearly_stats['Year'], y=yearly_stats['Success Rate'], 
                          name='Success Rate (%)', mode='lines+markers', 
                          line=dict(color='green', width=3)),
                secondary_y=True
            )
            fig.update_xaxes(title_text="Year")
            fig.update_yaxes(title_text="Total Launches", secondary_y=False)
            fig.update_yaxes(title_text="Success Rate (%)", secondary_y=True)
            fig.update_layout(title="Launch Activity and Success Rate Over Years", height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            st.dataframe(yearly_stats, use_container_width=True, hide_index=True)

def show_model_training(df):
    """Display the model training page"""
    st.title("🧠 Machine Learning Model Training")
    st.markdown("Train and evaluate multiple ML algorithms for rocket launch prediction")
    
    if df is not None:
        st.markdown("---")
        st.subheader("⚙️ Training Configuration")
        
        col1, col2 = st.columns(2)
        with col1:
            test_size = st.slider("Test Size (%)", min_value=10, max_value=40, value=20, step=5) / 100
        with col2:
            random_state = st.slider("Random State", min_value=1, max_value=100, value=42, step=1)
        
        st.markdown("---")
        st.subheader("📋 Select Models to Train")
        
        model_options = get_model_options()
        
        col1, col2, col3 = st.columns(3)
        selected_models = []
        
        for i, (model, model_type) in enumerate(model_options.items()):
            with [col1, col2, col3][i % 3]:
                if st.checkbox(f"{model}", value=True, key=f"model_{model}"):
                    selected_models.append(model)
                    st.caption(f"*{model_type}*")
        
        st.markdown("---")
        
        if st.button("▶️ Start Training", type="primary", use_container_width=True):
            if len(selected_models) == 0:
                st.warning("Please select at least one model to train")
            else:
                # Preprocess data
                with st.spinner("Preprocessing data..."):
                    X_train, X_test, y_train, y_test, scaler, label_encoders, feature_names, processed_df = preprocess_data(df, test_size, random_state)
                    st.session_state.X_train = X_train
                    st.session_state.X_test = X_test
                    st.session_state.y_train = y_train
                    st.session_state.y_test = y_test
                    st.session_state.scaler = scaler
                    st.session_state.label_encoders = label_encoders
                    st.session_state.feature_names = feature_names
                
                # Train models
                results, trained_models = train_models(X_train, X_test, y_train, y_test, selected_models)
                st.session_state.model_results = results
                st.session_state.trained_models = trained_models
                
                st.success(f"✅ Successfully trained {len(selected_models)} models!")
        
        # Display results if available
        if st.session_state.model_results:
            st.markdown("---")
            st.subheader("📊 Model Results")
            
            # Display model cards
            cols = st.columns(3)
            for idx, (model_name, metrics) in enumerate(st.session_state.model_results.items()):
                with cols[idx % 3]:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>{model_name}</h4>
                        <p><strong>Accuracy:</strong> {metrics['accuracy']*100:.2f}%</p>
                        <p><strong>Precision:</strong> {metrics['precision']*100:.2f}%</p>
                        <p><strong>Recall:</strong> {metrics['recall']*100:.2f}%</p>
                        <p><strong>F1-Score:</strong> {metrics['f1_score']*100:.2f}%</p>
                        <p><strong>Training Time:</strong> {metrics['training_time']:.2f}s</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Best model summary
            st.markdown("---")
            best_model = max(st.session_state.model_results.items(), 
                           key=lambda x: x[1]['accuracy'])
            st.success(f"🏆 Best Model: **{best_model[0]}** with {best_model[1]['accuracy']*100:.2f}% accuracy")

def show_model_comparison():
    """Display the model comparison page"""
    st.title("⚖️ Model Comparison Dashboard")
    st.markdown("Comprehensive analysis and comparison of ML model performance")
    
    if st.session_state.model_results:
        # Overview metrics
        st.subheader("📊 Performance Overview")
        
        metrics_list = []
        for model_name, metrics in st.session_state.model_results.items():
            metrics_list.append({
                'Model': model_name,
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1_score'],
                'Training Time': metrics['training_time']
            })
        
        metrics_df = pd.DataFrame(metrics_list)
        metrics_df = metrics_df.sort_values('Accuracy', ascending=False)
        
        # Display comparison table
        st.dataframe(
            metrics_df.style.format({
                'Accuracy': '{:.2%}',
                'Precision': '{:.2%}',
                'Recall': '{:.2%}',
                'F1-Score': '{:.2%}',
                'Training Time': '{:.3f}s'
            }).background_gradient(subset=['Accuracy'], cmap='RdYlGn'),
            use_container_width=True,
            hide_index=True
        )
        
        st.markdown("---")
        st.subheader("📊 Performance Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Performance comparison bar chart
            fig = go.Figure(data=[
                go.Bar(name='Accuracy', x=metrics_df['Model'], y=metrics_df['Accuracy']*100),
                go.Bar(name='Precision', x=metrics_df['Model'], y=metrics_df['Precision']*100),
                go.Bar(name='Recall', x=metrics_df['Model'], y=metrics_df['Recall']*100),
                go.Bar(name='F1-Score', x=metrics_df['Model'], y=metrics_df['F1-Score']*100)
            ])
            fig.update_layout(
                title='Performance Metrics Comparison',
                barmode='group',
                yaxis_title='Score (%)',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Best model summary
            best_model = metrics_df.iloc[0]
            st.markdown(f"""
            <div class="metric-card">
                <h3>🏆 Best Model: {best_model['Model']}</h3>
                <h2>{best_model['Accuracy']*100:.2f}%</h2>
                <p><strong>Accuracy</strong></p>
                <hr>
                <p><strong>Precision:</strong> {best_model['Precision']*100:.2f}%</p>
                <p><strong>Recall:</strong> {best_model['Recall']*100:.2f}%</p>
                <p><strong>F1-Score:</strong> {best_model['F1-Score']*100:.2f}%</p>
                <p><strong>Training Time:</strong> {best_model['Training Time']:.3f}s</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("⚠️ No trained models available. Please train models first in the Model Training page.")
