import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Digital Distraction Detector", layout="centered")

st.title("📱 Digital Distraction Detector")
st.write("Supervised Learning Classification Project")

# -------------------------
# Load Dataset
# -------------------------
@st.cache_data
def load_data():
    return pd.read_csv("data.csv")

data = load_data()

# Encode labels
le = LabelEncoder()
data["productivity_label"] = le.fit_transform(data["productivity_label"])

X = data.drop("productivity_label", axis=1)
y = data["productivity_label"]

# Train Model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Model Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.subheader("📊 Model Accuracy")
st.success(f"Accuracy: {accuracy * 100:.2f}%")

# -------------------------
# User Input Section
# -------------------------
st.subheader("🔎 Enter Your Daily Data")

total_screen_time = st.slider("Total Screen Time (hrs)", 0, 15, 6)
social_media_time = st.slider("Social Media Time (hrs)", 0, 10, 2)
study_time = st.slider("Study Time (hrs)", 0, 10, 4)
breaks_taken = st.slider("Number of Breaks Taken", 0, 15, 4)
sleep_hours = st.slider("Sleep Hours", 0, 12, 7)

if st.button("Predict Productivity"):
    input_data = np.array([[total_screen_time,
                            social_media_time,
                            study_time,
                            breaks_taken,
                            sleep_hours]])

    prediction = model.predict(input_data)
    result = le.inverse_transform(prediction)

    st.subheader("🎯 Prediction Result")
    st.success(result[0])

# -------------------------
# Confusion Matrix
# -------------------------
st.subheader("📉 Confusion Matrix")

cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
plt.xlabel("Predicted")
plt.ylabel("Actual")
st.pyplot(fig)

# -------------------------
# Feature Importance
# -------------------------
st.subheader("📌 Feature Importance")

importance = model.feature_importances_
feature_names = X.columns

fig2, ax2 = plt.subplots()
ax2.barh(feature_names, importance)
st.pyplot(fig2)