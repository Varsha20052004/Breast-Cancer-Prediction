from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import shap

# Load the dataset
data = pd.read_csv('Breast_cancer_data.csv')

# Separate features and target
X = data.drop('diagnosis', axis=1)
y = data['diagnosis']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build the neural network model
model = Sequential([
    Dense(16, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train_scaled, y_train, epochs=50, batch_size=16, validation_split=0.2, verbose=1)

# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test_scaled, y_test)
print(f'Test Accuracy: {accuracy:.4f}')

# Create SHAP explainer using KernelExplainer
explainer = shap.KernelExplainer(lambda x: model.predict(x).flatten(), X_train_scaled[:100])

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    final_features = np.array(features).reshape(1, -1)
    final_features_scaled = scaler.transform(final_features)
    prediction = model.predict(final_features_scaled)[0][0]
    output = 'Malignant' if prediction > 0.5 else 'Benign'
    
    # Compute SHAP values for the input
    shap_values = explainer.shap_values(final_features_scaled, nsamples=100)
    
    return render_template('index.html', prediction_text=f'Cancer prediction: {output}', shap_values=shap_values[0].tolist())

@app.route('/shap_summary', methods=['GET'])
def shap_summary():
    shap_values = np.array(request.args.get('shap_values'))
    shap.summary_plot(shap_values, feature_names=X.columns)
    return "SHAP summary plot generated."

if __name__ == "__main__":
    app.run(debug=True)
