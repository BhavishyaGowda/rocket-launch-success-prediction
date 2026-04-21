# 🚀 Rocket Launch Success Prediction Dashboard

A machine learning-powered web application that predicts the success probability of rocket launches using historical data and advanced ML algorithms.

## 🌟 Features

- **Interactive Dashboard**: Visualize launch statistics and trends
- **Multiple ML Models**: Train and compare different algorithms:
  - Logistic Regression
  - K-Nearest Neighbors
  - Naive Bayes
  - Neural Network
  - XGBoost
- **Real-time Predictions**: Input launch parameters and get instant predictions
- **Bulk Predictions**: Upload CSV files for batch predictions
- **Data Export**: Download results in CSV or Excel format
- **Secure Access**: User authentication system

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/rocket-launch-prediction.git
cd rocket-launch-prediction
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🚀 Usage

1. Start the Streamlit app:
```bash
streamlit run streamlit_app.py
```python -m streamlit run streamlit_app.py


2. Access the dashboard:
- Open your browser and go to `http://localhost:8501`
- Login using the provided credentials:
  - Username: `Sameena`
  - Password: `sam@7619`

3. Navigate through the sections:
- **Dashboard**: View launch statistics and trends
- **Model Training**: Train ML models on the dataset
- **Model Comparison**: Compare performance of different models
- **Predictions**: Make new launch predictions

## 📊 Project Structure

```
├── streamlit_app.py   # Main application entry point
├── auth.py           # Authentication module
├── utils.py          # Data processing utilities
├── ml_models.py      # Machine learning models
├── pages.py          # Dashboard pages
├── predictions.py    # Prediction interface
├── requirements.txt  # Project dependencies
└── Documentation.md  # Detailed documentation
```

## 🔧 Technologies Used

- **Frontend**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-learn, XGBoost
- **Visualization**: Plotly
- **Data Export**: OpenPyXL

## 📈 Model Performance

The system uses multiple ML models and ensemble voting for predictions:

| Model | Typical Accuracy | Use Case |
|-------|-----------------|-----------|
| Logistic Regression | 85-90% | Baseline predictions |
| KNN | 82-88% | Pattern matching |
| Naive Bayes | 80-85% | Probabilistic approach |
| Neural Network | 87-92% | Complex patterns |
| XGBoost | 88-93% | Ensemble learning |

## 📥 Input Parameters

Required parameters for launch prediction:
- Company name
- Rocket type
- Launch date & time
- Launch site & location
- Orbit type & destination
- Mission & payload details
- Geographic coordinates
- Weather conditions

## 📤 Output Format

Predictions include:
- Success/Failure prediction
- Confidence score
- Individual model predictions
- Historical success rates
- Visualization of results

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **Bhavishya P T**

## 🙏 Acknowledgments

- Historical launch data providers
- Streamlit community
- Machine learning libraries contributors
