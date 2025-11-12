import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Title and Description
st.set_page_config(page_title="Student Score Predictor", page_icon="🎓", layout="centered")
st.title("🎓 AI-Based Student Score Prediction and Analysis System")
st.write("This app predicts student scores based on study hours and provides performance insights.")

# Load model and dataset
model = joblib.load("student_score_model.pkl")
data = pd.read_csv("student_scores.csv")

# Sidebar Navigation
option = st.sidebar.radio("Select Mode:", ["📈 Predict Score", "📊 Analyze Student Performance"])

# --- MODE 1: Prediction ---
if option == "📈 Predict Score":
    st.subheader("📘 Dataset Preview")
    st.write(data.head())

    st.subheader("📊 Regression Line (Hours vs Scores)")
    fig, ax = plt.subplots()
    ax.scatter(data['Hours'], data['Scores'], color='blue', label='Actual Data')
    ax.plot(data['Hours'], model.predict(data[['Hours']]), color='red', label='Regression Line')
    ax.set_xlabel('Hours Studied')
    ax.set_ylabel('Score')
    ax.legend()
    st.pyplot(fig)

    hours = st.number_input("Enter study hours:", min_value=0.0, max_value=12.0, step=0.5)
    if st.button("Predict Score"):
        score = model.predict([[hours]])[0]
        st.success(f"📈 Predicted Score: {score:.2f}")

# --- MODE 2: Performance Analysis ---
elif option == "📊 Analyze Student Performance":
    st.subheader("🧠 Student Performance Analyzer")
    score_input = st.number_input("Enter Actual Score:", min_value=0.0, max_value=100.0, step=1.0)
    hours_input = st.number_input("Enter Hours Studied:", min_value=0.0, max_value=12.0, step=0.5)

    if st.button("Analyze Performance"):
        if hours_input == 0:
            st.warning("Hours cannot be zero for efficiency calculation!")
        else:
            efficiency = score_input / hours_input
            if efficiency < 8:
                category = "❌ Needs Improvement"
                suggestion = "Increase study hours and practice daily."
            elif 8 <= efficiency < 12:
                category = "✅ Average Performer"
                suggestion = "Maintain your consistency and improve time management."
            else:
                category = "🌟 Excellent Performer"
                suggestion = "Keep up the great work and help peers too!"

            st.write(f"**Efficiency (Score/Hour):** {efficiency:.2f}")
            st.write(f"**Category:** {category}")
            st.info(f"Recommendation: {suggestion}")
