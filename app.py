import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Load the trained model
with open("student_performance_model.pkl", "rb") as file:
    model = pickle.load(file)

# Page configuration
st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓")

# Title
st.title("🎓 Student Performance Predictor")
st.write("Predict a student's final marks using Machine Learning.")

# Sidebar
st.sidebar.title("ℹ️ About")
st.sidebar.info(
    "This ML app predicts student performance using Linear Regression."
)

# User Inputs
study_hours = st.slider("Study Hours", 0.0, 12.0, 5.0)
attendance = st.slider("Attendance (%)", 0, 100, 75)
previous_marks = st.slider("Previous Marks", 0, 100, 60)

# Performance classification function
def classify_performance(score):
    if score >= 80:
        return "Excellent 🌟"
    elif score >= 50:
        return "Average 👍"
    else:
        return "Needs Improvement 📘"

# Prediction Button
if st.button("Predict Performance"):
    input_data = np.array([[study_hours, attendance, previous_marks]])
    prediction = model.predict(input_data)[0]
    category = classify_performance(prediction)

    st.subheader("📊 Prediction Result")
    st.success(f"Predicted Final Marks: {prediction:.2f}")
    st.info(f"Performance Category: {category}")

    # Progress Bar
    st.write("### 📈 Performance Score")
    st.progress(min(int(prediction), 100))

    # Bar Chart Visualization
    input_df = pd.DataFrame({
        "Feature": ["Study Hours", "Attendance", "Previous Marks"],
        "Value": [study_hours, attendance, previous_marks]
    })

    st.subheader("📊 Input Summary")
    st.bar_chart(input_df.set_index("Feature"))

# Footer
st.markdown("---")
st.markdown("Developed by **Aditya Chauhan** | BTech CSE Student")