"""
Authentication module for Rocket Launch Prediction Dashboard
"""
import streamlit as st
from pathlib import Path

# Authentication credentials
VALID_CREDENTIALS = {
    "Sameena.M.M.": "sam@7619"
}

# Read your HTML file
with open("homepage.html","r",encoding="utf-8") as f:
    html_content=f.read()

#Inject it into streamlit app
st.components.v1.html(html_content,height=800,scrolling=True)

def show_login_page():
    """Display the login page with homepage content"""
    
    st.markdown("### 🚀 Rocket Launch Success Prediction")
    st.markdown("Please enter your credentials to access the rocket launch prediction dashboard.")
    
    with st.form("login_form"):
        username = st.text_input("Username", placeholder="Enter username")
        password = st.text_input("Password", type="password", placeholder="Enter password")
        submit = st.form_submit_button("🚀 Login", use_container_width=True)
        
        if submit:
            if username in VALID_CREDENTIALS and VALID_CREDENTIALS[username] == password:
                st.session_state.authenticated = True
                st.session_state.username = username
                st.session_state.show_homepage = False
                st.success(f"Welcome, {username}! 🎉")
                st.rerun()
            else:
                st.error("❌ Invalid username or password")
                st.info("💡 Hint: Check your credentials and try again")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; color: #718096; padding: 20px;">
            <p>🚀 Rocket Launch Prediction System | Machine Learning Dashboard</p>
            <p style="font-size: 0.9rem;">Powered by Streamlit, scikit-learn, and XGBoost</p>
        </div>
    """, unsafe_allow_html=True)

def logout():
    """Logout function"""
    st.session_state.authenticated = False
    st.session_state.username = None
    st.session_state.show_homepage = True
    st.rerun()

def init_session_state():
    """Initialize all session state variables"""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'username' not in st.session_state:
        st.session_state.username = None
    if 'show_homepage' not in st.session_state:
        st.session_state.show_homepage = True
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'trained_models' not in st.session_state:
        st.session_state.trained_models = {}
    if 'model_results' not in st.session_state:
        st.session_state.model_results = {}
    if 'scaler' not in st.session_state:
        st.session_state.scaler = None
    if 'label_encoders' not in st.session_state:
        st.session_state.label_encoders = {}
    if 'feature_names' not in st.session_state:
        st.session_state.feature_names = []
    if 'X_train' not in st.session_state:
        st.session_state.X_train = None
    if 'X_test' not in st.session_state:
        st.session_state.X_test = None
    if 'y_train' not in st.session_state:
        st.session_state.y_train = None
    if 'y_test' not in st.session_state:
        st.session_state.y_test = None
