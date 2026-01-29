import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

st.title("🤖 Random Forest ML App (CSV Dataset)")

# ------------------ Load Dataset ------------------
data = pd.read_csv("RF datasets.csv")

st.write("### Dataset Preview")
st.dataframe(data.head())

# ------------------ Encode Categorical Columns ------------------
label_encoders = {}

for col in data.columns:
    if data[col].dtype == 'object':
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le

# ------------------ Split Features & Target ------------------
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# ------------------ Train Model ------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)

# ------------------ User Input ------------------
st.write("## Enter Feature Values")

user_input = []

for col in X.columns:
    value = st.number_input(f"{col}", float(X[col].min()), float(X[col].max()))
    user_input.append(value)

# ------------------ Prediction ------------------
if st.button("Predict"):
    input_array = np.array([user_input])
    prediction = model.predict(input_array)[0]

    # Decode prediction if target was categorical
    target_col = data.columns[-1]
    if target_col in label_encoders:
        prediction = label_encoders[target_col].inverse_transform([prediction])[0]

    st.success(f"Prediction: **{prediction}**")

# ------------------ Sidebar ------------------
st.sidebar.header("Model Info")
st.sidebar.write("Algorithm: Random Forest")
st.sidebar.write(f"Accuracy: **{accuracy:.2f}**")
st.sidebar.write(f"Features: {X.shape[1]}")