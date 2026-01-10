import streamlit as st
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv # pyright: ignore[reportMissingImports]
from openai import OpenAI # type: ignore

# Scikit-learn imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# --- 1. Setup & Configuration ---
st.set_page_config(page_title="CalmAI Stress Detection", page_icon="🤖", layout="centered")

# Load environment variables for OpenAI
load_dotenv()

# Initialize OpenAI Client
# Ensure you have a .env file with OPENAI_API_KEY=sk-...
api_key = os.getenv("OPENAI_API_KEY")

# --- 2. Data Loading & Caching ---
@st.cache_data                      
def load_data():
    current_directory = os.path.dirname(os.path.abspath(__file__))
    file_full_path = os.path.join(current_directory, 'stress_data.csv')
    
    try:
        df = pd.read_csv(file_full_path)
    except FileNotFoundError:
        return None
    
    if 'strees_level' in df.columns:
        df.rename(columns={'strees_level': 'stress_level'}, inplace=True)
        
    return df

# --- 3. Model Training & Caching ---
@st.cache_resource
def train_models(df):
    X = df[['BPM', 'GSR']]
    y = df['stress_level']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    models = {}
    scores = {}

    # SVM
    svm = SVC(kernel='linear', random_state=42)
    svm.fit(X_train, y_train)
    models['SVM (Linear)'] = svm
    scores['SVM (Linear)'] = accuracy_score(y_test, svm.predict(X_test))

    # KNN
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    models['KNN'] = knn
    scores['KNN'] = accuracy_score(y_test, knn.predict(X_test))

    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    models['Random Forest'] = rf
    scores['Random Forest'] = accuracy_score(y_test, rf.predict(X_test))

    # Gradient Boosting
    gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    gb.fit(X_train, y_train)
    models['Gradient Boosting'] = gb
    scores['Gradient Boosting'] = accuracy_score(y_test, gb.predict(X_test))

    return models, scores, scaler

# --- 4. Helper Function: Get AI Advice ---
def get_stress_advice():
    """Fetches advice from OpenAI if stress is high."""
    if not api_key:
        return "⚠️ OpenAI API Key not found. Please check your .env file."
    
    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini", # Updated to a valid standard model name
            messages=[
                {"role": "system", "content": "You are a psychologist who helps people reduce stress and improve mental health."},
                {"role": "user", "content": "The user has just been detected with high stress levels based on physiological data. Provide tips to reduce stress."}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error connecting to AI: {e}"

# --- 5. Main App Execution ---
df = load_data()

if df is None:
    st.error("Error: 'stress_data.csv' not found. Please ensure the dataset is in the same folder as app.py.")
else:
    models, scores, scaler = train_models(df)
    
    st.title("🤖 CalmAi Stress Prediction & Relief App")
    st.write("CalmAi is a machine learning-powered stress prediction and relief assistant. Please do note that this is soley a wellness device that is not intended for medical use. We collect your heart beats and GSR readings to predict stress levels. Using multiple ML models, we provide a consensus stress level and personalized advice if high stress is detected. If stress is high, CalmAi offers tailored recommendations to help you relax and improve your mental well-being. Please enter your Sensor data below:")

    col1, col2 = st.columns(2)
    with col1:
        bpm_text = st.text_input("Heart Rate (BPM)", value="0")
    with col2:
        gsr_text = st.text_input("GSR Reading", value="0")

    if st.button("Predict Stress Level", type="primary"):
        try:
            # Data Pre-processing
            bpm_value = float(bpm_text)
            gsr_value = float(gsr_text)
            input_data = np.array([[bpm_value, gsr_value]])
            input_scaled = scaler.transform(input_data)

            # Run Predictions
            results = []
            for name, model in models.items():
                prediction = model.predict(input_scaled)[0]
                accuracy = scores[name]
                pred_label = "High Stress (1)" if prediction == 1 else "Low Stress (0)"
                results.append({"Model": name, "Accuracy": f"{accuracy:.2%}", "Prediction": pred_label})

            results_df = pd.DataFrame(results)
            st.write("### Analysis Results")
            st.dataframe(results_df, use_container_width=True)

            # Consensus Logic
            high_stress_count = results_df[results_df['Prediction'].str.contains("High")].shape[0]
            low_stress_count = results_df[results_df['Prediction'].str.contains("Low")].shape[0]

            if high_stress_count > low_stress_count:
                st.error(f"⚠️ Consensus: **High Stress** detected.")
                
                # --- AI INTERVENTION START ---
                st.markdown("---")
                st.subheader("🧠 CalmAi Recommendation")
                with st.spinner("Generating personalized stress relief advice..."):
                    advice = get_stress_advice()
                    st.info(advice)
                # --- AI INTERVENTION END ---

            elif low_stress_count > high_stress_count:
                st.success(f"✅ Consensus: **Low Stress** detected. Keep it up!")
            else:
                st.warning("⚖️ Consensus: Models are tied.")

        except ValueError:

            st.error("❌ Invalid Input: Please enter numbers only.")
