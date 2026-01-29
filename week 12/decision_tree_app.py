import streamlit as st
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# ------------------ Load Dataset ------------------
iris = load_iris()
X, y = iris.data, iris.target

# ------------------ Train Model ------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# ------------------ Streamlit UI ------------------
st.title("🌸 Iris Flower Species Predictor (Decision Tree)")
st.write("Enter flower measurements to predict species")

# Inputs
sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.4)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.4)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 1.3)
petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

# Prediction
if st.button("Predict Species"):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)[0]
    species = iris.target_names[prediction]

    st.success(f"Predicted Species: **{species.capitalize()}**")

# ------------------ Show Model Accuracy ------------------
accuracy = model.score(X_test, y_test)
st.sidebar.header("📊 Model Info")
st.sidebar.write(f"Accuracy: **{accuracy:.2f}**")