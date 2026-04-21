"""
Rocket Launch Success Prediction Dashboard
Main application entry point
"""
import streamlit as st

# Import custom modules
from auth import show_login_page, logout, init_session_state
from utils import load_data
from pages import show_dashboard, show_model_training, show_model_comparison
from predictions import show_predictions

# Page configuration
st.set_page_config(
    page_title="🚀 Rocket Launch Prediction Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {        
        padding: 10px;
        border-radius: 5px;
    }
    .metric-card {
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    .login-container {
        max-width: 400px;
        margin: 50px auto;
        padding: 40px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.3);
    }
    .homepage-content {
        text-align: center;
        padding: 20px;
        background: rgba(17, 24, 39, 0.6);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        margin-bottom: 30px;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
init_session_state()

# Check authentication status
if not st.session_state.authenticated:
    show_login_page()
    st.stop()

# Sidebar
with st.sidebar:
    # User info and logout
    st.markdown(f"### 👤 Welcome, {st.session_state.username}!")
    if st.button("🚪 Logout", use_container_width=True):
        logout()
    
    st.markdown("---")
    
    st.title("🚀 Navigation")
    page = st.radio("Select Page", 
                   ["📊 Dashboard", "🧠 Model Training", "⚖️ Model Comparison", "🔮 Predictions"],
                   label_visibility="collapsed")
    
    st.markdown("---")
    st.markdown("### About")
    st.info("This dashboard provides comprehensive analytics and machine learning capabilities for rocket launch prediction.")

# Load data
if st.session_state.data is None:
    st.session_state.data = load_data()

df = st.session_state.data

# Main content based on selected page
if page == "📊 Dashboard":
    show_dashboard(df)

elif page == "🧠 Model Training":
    show_model_training(df)

elif page == "⚖️ Model Comparison":
    show_model_comparison()

elif page == "🔮 Predictions":
    show_predictions(df)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 20px;'>
    <p>🚀 Rocket Launch Prediction Dashboard | Built with Streamlit</p>
    <p>Data-driven space launch analytics with machine learning</p>
</div>
""", unsafe_allow_html=True)