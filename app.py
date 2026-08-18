import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64
from PIL import Image
import io

# Page configuration
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stButton > button {
        width: 100%;
        background-color: #ff4b4b;
        color: white;
        font-size: 20px;
        font-weight: bold;
        border-radius: 10px;
        border: none;
        padding: 10px 24px;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        background-color: #ff6b6b;
        transform: scale(1.02);
        box-shadow: 0 4px 12px rgba(255, 75, 75, 0.3);
    }
    .stButton > button:active {
        transform: scale(0.98);
    }
    .prediction-box {
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        margin: 10px 0;
        animation: fadeIn 0.5s;
    }
    .prediction-positive {
        background: linear-gradient(135deg, #ff6b6b, #ee5a24);
        color: white;
        box-shadow: 0 4px 15px rgba(238, 90, 36, 0.3);
    }
    .prediction-negative {
        background: linear-gradient(135deg, #00b894, #00a86b);
        color: white;
        box-shadow: 0 4px 15px rgba(0, 184, 148, 0.3);
    }
    .feature-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin: 10px 0;
        border-left: 4px solid #ff4b4b;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    .header-container {
        background: linear-gradient(135deg, #ff4b4b 0%, #c0392b 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .info-box {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Load models and data
@st.cache_resource
def load_models():
    try:
        model = joblib.load('LR_heart.pkl')
        scaler = joblib.load('scaler.pkl')
        columns = joblib.load('columns.pkl')
        return model, scaler, columns
    except FileNotFoundError as e:
        st.error(f"❌ Model files not found. Please ensure the following files are in the same directory:\n- LR_heart.pkl\n- scaler.pkl\n- columns.pkl")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading model files: {str(e)}")
        st.stop()

model, scaler, feature_columns = load_models()

# Header
st.markdown("""
    <div class="header-container">
        <h1 style="font-size: 48px; margin-bottom: 10px;">❤️ Heart Disease Prediction</h1>
        <p style="font-size: 20px; opacity: 0.9;">AI-powered early detection system for cardiovascular health</p>
    </div>
""", unsafe_allow_html=True)

# Sidebar for input
with st.sidebar:
    st.markdown("### 🏥 Patient Information")
    st.markdown("---")
    
    # Create input fields with better formatting
    age = st.slider("📅 Age", 18, 100, 50, help="Age in years")
    
    resting_bp = st.slider("💓 Resting Blood Pressure", 80, 200, 120, 
                          help="Resting blood pressure in mm Hg")
    
    cholesterol = st.slider("🩸 Cholesterol Level", 100, 400, 200,
                           help="Serum cholesterol in mg/dl")
    
    fasting_bs = st.radio("🍽️ Fasting Blood Sugar > 120 mg/dl", 
                         ["No", "Yes"], index=0,
                         help="Fasting blood sugar level")
    
    max_hr = st.slider("🏃 Maximum Heart Rate Achieved", 60, 220, 150,
                      help="Maximum heart rate during exercise")
    
    oldpeak = st.slider("📊 ST Depression (Oldpeak)", 0.0, 6.0, 1.0, 0.1,
                       help="ST depression induced by exercise relative to rest")
    
    st.markdown("---")
    st.markdown("### 🎯 Additional Risk Factors")
    
    sex = st.radio("⚧️ Sex", ["Male", "Female"])
    
    chest_pain = st.selectbox("🫀 Chest Pain Type", 
                             ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"])
    
    resting_ecg = st.selectbox("📈 Resting ECG Results", 
                              ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"])
    
    exercise_angina = st.radio("🏋️ Exercise Induced Angina", ["No", "Yes"])
    
    st_slope = st.selectbox("📉 ST Slope", ["Upsloping", "Flat", "Downsloping"])
    
    st.markdown("---")
    
    # Predict button
    predict_button = st.button("🫀 Predict Heart Disease", use_container_width=True)

# Main content area
if predict_button:
    try:
        # Prepare input data
        sex_m = 1 if sex == "Male" else 0
        
        chest_pain_map = {
            "Typical Angina": "ATA",
            "Atypical Angina": "NAP",
            "Non-Anginal Pain": "TA",
            "Asymptomatic": "ASY"
        }
        chest_pain_type = chest_pain_map[chest_pain]
        
        resting_ecg_map = {
            "Normal": "Normal",
            "ST-T Wave Abnormality": "ST",
            "Left Ventricular Hypertrophy": "LVH"
        }
        resting_ecg_type = resting_ecg_map[resting_ecg]
        
        exercise_angina_y = 1 if exercise_angina == "Yes" else 0
        st_slope_flat = 1 if st_slope == "Flat" else 0
        st_slope_up = 1 if st_slope == "Upsloping" else 0
        
        # Create feature dictionary with all required columns
        input_dict = {
            'Age': age,
            'RestingBP': resting_bp,
            'Cholesterol': cholesterol,
            'FastingBS': 1 if fasting_bs == "Yes" else 0,
            'MaxHR': max_hr,
            'Oldpeak': oldpeak,
            'Sex_M': sex_m,
            'ChestPainType_ATA': 1 if chest_pain_type == "ATA" else 0,
            'ChestPainType_NAP': 1 if chest_pain_type == "NAP" else 0,
            'ChestPainType_TA': 1 if chest_pain_type == "TA" else 0,
            'RestingECG_Normal': 1 if resting_ecg_type == "Normal" else 0,
            'RestingECG_ST': 1 if resting_ecg_type == "ST" else 0,
            'ExerciseAngina_Y': exercise_angina_y,
            'ST_Slope_Flat': st_slope_flat,
            'ST_Slope_Up': st_slope_up
        }
        
        # Convert to DataFrame
        input_df = pd.DataFrame([input_dict])
        
        # Ensure columns are in correct order
        input_df = input_df[feature_columns]
        
        # Scale features
        input_scaled = scaler.transform(input_df)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0]
        
        # Display results in columns
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            # Prediction result
            if prediction == 1:
                st.markdown("""
                    <div class="prediction-box prediction-positive">
                        <h2 style="font-size: 36px;">⚠️ High Risk</h2>
                        <p style="font-size: 20px;">Heart disease detected with high probability</p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                    <div class="prediction-box prediction-negative">
                        <h2 style="font-size: 36px;">✅ Low Risk</h2>
                        <p style="font-size: 20px;">No significant heart disease indicators found</p>
                    </div>
                """, unsafe_allow_html=True)
        
        # Probability visualization
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Prediction Probability")
            
            # Create gauge chart using plotly
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = probability[1] * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Risk Probability (%)"},
                delta = {'reference': 50, 'increasing': {'color': "red"}},
                gauge = {
                    'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': "red" if probability[1] > 0.5 else "green"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 30], 'color': 'lightgreen'},
                        {'range': [30, 60], 'color': 'lightyellow'},
                        {'range': [60, 100], 'color': 'lightcoral'}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            
            fig.update_layout(
                height=300,
                margin=dict(l=20, r=20, t=50, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(size=16)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 📈 Risk Factors Analysis")
            
            # Create bar chart for risk factors
            risk_factors = {
                'Age': age,
                'Blood Pressure': resting_bp,
                'Cholesterol': cholesterol,
                'Fasting Blood Sugar': 1 if fasting_bs == "Yes" else 0,
                'Max Heart Rate': max_hr,
                'ST Depression': oldpeak
            }
            
            risk_df = pd.DataFrame({
                'Factor': list(risk_factors.keys()),
                'Value': list(risk_factors.values())
            })
            
            # Normalize values for better visualization
            risk_df['Normalized'] = risk_df['Value'] / risk_df['Value'].max()
            
            fig2 = go.Figure(data=[
                go.Bar(
                    x=risk_df['Factor'],
                    y=risk_df['Value'],
                    text=risk_df['Value'],
                    textposition='auto',
                    marker_color=['#ff6b6b' if i in [0, 1, 2] else '#667eea' for i in range(len(risk_df))],
                    marker_line_color='rgb(8,48,107)',
                    marker_line_width=1.5,
                    opacity=0.8
                )
            ])
            
            fig2.update_layout(
                title="Risk Factor Values",
                xaxis_title="Factors",
                yaxis_title="Value",
                height=300,
                margin=dict(l=20, r=20, t=50, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )
            
            st.plotly_chart(fig2, use_container_width=True)
        
        # Detailed analysis
        st.markdown("---")
        st.markdown("### 📋 Detailed Risk Assessment")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Age", f"{age} years", 
                     "High risk" if age > 60 else "Low risk")
        
        with col2:
            st.metric("Blood Pressure", f"{resting_bp} mm Hg",
                     "Elevated" if resting_bp > 130 else "Normal")
        
        with col3:
            st.metric("Cholesterol", f"{cholesterol} mg/dl",
                     "High" if cholesterol > 200 else "Normal")
        
        with col4:
            st.metric("Heart Rate", f"{max_hr} bpm",
                     "Good" if max_hr > 140 else "Low")
        
        # Recommendations
        st.markdown("---")
        st.markdown("### 💡 Health Recommendations")
        
        recommendations = []
        if age > 60:
            recommendations.append("🏥 Regular cardiovascular screening recommended")
        if resting_bp > 130:
            recommendations.append("🩺 Monitor blood pressure and consider lifestyle changes")
        if cholesterol > 200:
            recommendations.append("🥗 Review diet and consider cholesterol-lowering strategies")
        if fasting_bs == "Yes":
            recommendations.append("🍬 Monitor blood sugar levels and consult with healthcare provider")
        if oldpeak > 2.0:
            recommendations.append("💊 High ST depression detected - consult cardiologist")
        
        if recommendations:
            for rec in recommendations:
                st.info(rec)
        else:
            st.success("✅ All vital signs are within healthy ranges. Maintain your current lifestyle!")
            
    except Exception as e:
        st.error(f"❌ An error occurred during prediction: {str(e)}")
        st.info("Please check all inputs and try again.")

else:
    # Welcome message with information
    st.markdown("""
    <div style="padding: 20px;">
        <h2>Welcome to the Heart Disease Prediction Tool</h2>
        <p style="font-size: 18px; color: #555;">
            This AI-powered tool uses machine learning to assess the risk of heart disease
            based on various health parameters. Fill in the patient information in the sidebar
            and click the predict button to get started.
        </p>
        
        <div style="display: flex; gap: 20px; margin-top: 30px; flex-wrap: wrap;">
            <div class="feature-card" style="flex: 1; min-width: 200px;">
                <h3>🔬 AI-Powered</h3>
                <p>Using advanced machine learning algorithms for accurate predictions</p>
            </div>
            <div class="feature-card" style="flex: 1; min-width: 200px;">
                <h3>⚡ Real-Time</h3>
                <p>Instant analysis and visualization of your health data</p>
            </div>
            <div class="feature-card" style="flex: 1; min-width: 200px;">
                <h3>📊 Comprehensive</h3>
                <p>Detailed risk assessment with actionable insights</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)