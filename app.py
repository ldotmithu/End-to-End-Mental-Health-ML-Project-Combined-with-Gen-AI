import streamlit as st
import pandas as pd
import time
import logging
from io import StringIO
from src.ml_ganAI.gen_ai.helper import GenAI
from src.ml_ganAI.Pipeline.prediction import Predication_Pipeline
import os 


gen_ai = GenAI()
prediction_pipeline = Predication_Pipeline()


st.markdown(
    """
    <style>
    .main {
        background-color: #f5f5f5;
        padding: 20px;
        border-radius: 10px;
    }
    .sidebar .sidebar-content {
        background-color: #ffffff;
    }
    h1 {
        color: #4a90e2;
    }
    h2 {
        color: #4a90e2;
    }
    .stButton button {
        background-color: #4a90e2;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 16px;
    }
    .stButton button:hover {
        background-color: #357abd;
    }
    .footer {
        text-align: center;
        padding: 10px;
        margin-top: 20px;
        font-size: 14px;
    }
    .footer a {
        color: #4a90e2;
        text-decoration: none;
    }
    .footer a:hover {
        text-decoration: underline;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


st.title("🧠 MindCare: AI-Powered Mental Health Prediction System")


st.sidebar.header("📊 Input Your Health Data")


age = st.sidebar.number_input("Age", min_value=0, max_value=120, value=25)
systolic_bp = st.sidebar.number_input("Systolic BP", min_value=0, max_value=300, value=120)
diastolic_bp = st.sidebar.number_input("Diastolic BP", min_value=0, max_value=300, value=80)
bs = st.sidebar.number_input("Blood Sugar (BS)", min_value=0, max_value=500, value=90)
body_temp = st.sidebar.number_input("Body Temp", min_value=0.0, max_value=120.0, value=98.6)
heart_rate = st.sidebar.number_input("Heart Rate", min_value=0, max_value=300, value=72)

input_data = {
    "Age": [age],
    "SystolicBP": [systolic_bp],
    "DiastolicBP": [diastolic_bp],
    "BS": [bs],
    "BodyTemp": [body_temp],
    "HeartRate": [heart_rate],
}
input_data_df = pd.DataFrame(input_data)


# st.subheader("📋 Your Input Data")
# st.dataframe(input_data_df.style.highlight_max(axis=0), use_container_width=True)


if st.sidebar.button("🚀 Predict"):
    with st.spinner("🔮 Predicting your mental health status..."):
        try:
            
            input_data_processed = prediction_pipeline.transform(input_data_df)
            prediction = prediction_pipeline.prediction(input_data_processed)
            if prediction == 2:
                prediction = "High Risk"
            elif prediction == 1:    
                prediction = "Mid Risk"
            else:
                prediction = "Low Risk"    

           
            response = gen_ai.response(input_data=input_data_df.values.tolist())

            st.subheader("🎯 Prediction Result")
            st.success(f"**Prediction:** {prediction}")

            st.subheader("💡 AI Suggestions")
            st.info(response)

        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")


st.sidebar.header("🛠️ Model Training")

if st.sidebar.button("🔧 Train Model"):
    with st.spinner("Training in progress..."):
        try:
            os.system("python main.py")
            st.success("🎉 Training completed successfully!")
            
        except Exception as e:
            st.error(f"❌ Training failed: {str(e)}")


st.markdown("---")
st.markdown(
    """
    <div class="footer">
        <p>🛠️ Built with ❤️ by <strong>ldotmithu</strong></p>
        <p>
            <a href="https://github.com/your-github-profile" target="_blank">
                <img src="https://img.icons8.com/ios-filled/50/000000/github.png" width="20" height="20"/>
                GitHub
            </a>
            &nbsp;|&nbsp;
            <a href="https://www.linkedin.com/in/your-linkedin-profile" target="_blank">
                <img src="https://img.icons8.com/ios-filled/50/000000/linkedin.png" width="20" height="20"/>
                LinkedIn
            </a>
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown("---")