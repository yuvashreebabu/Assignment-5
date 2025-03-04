import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle
import streamlit as st

# Step 1: Load Dataset
iris = datasets.load_iris()
data = pd.DataFrame(iris.data, columns=iris.feature_names)
data['species'] = iris.target

# Step 2: Preprocess Data
X = data.iloc[:, :-1]
y = data['species']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into Training and Testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 3: Train Multi-Class Logistic Regression Model
model = LogisticRegression(multi_class='ovr', max_iter=200)
model.fit(X_train, y_train)

# Model Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)

# Save Model
with open('iris_model.pkl', 'wb') as file:
    pickle.dump(model, file)
    pickle.dump(scaler, file)

# Streamlit App
def main():
    st.title("Iris Flower Species Prediction")
    st.write("Enter the sepal and petal dimensions to predict the species.")
    
    sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, step=0.1)
    sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, step=0.1)
    petal_length = st.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, step=0.1)
    petal_width = st.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, step=0.1)

    if st.button("Predict"):
        with open('iris_model.pkl', 'rb') as file:
            loaded_model = pickle.load(file)
            loaded_scaler = pickle.load(file)
            input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
            input_scaled = loaded_scaler.transform(input_data)
            prediction = loaded_model.predict(input_scaled)
            st.success(f"Predicted Species: {iris.target_names[prediction[0]]}")

if __name__ == '__main__':
    main()
