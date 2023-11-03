import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf

PATH = 'Crop_recommendation.csv'
data = pd.read_csv(PATH)

input_data = data[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
output_crop = data['label']

X_train, X_test, Y_train, Y_test = train_test_split(input_data, output_crop, test_size=0.2, random_state=42)

LogReg = LogisticRegression(random_state=42).fit(X_train, Y_train)

new_input = np.array([[67, 41, 40, 25.848795, 87.81661683, 7.333143205, 152.6194403]])
predicted_values = LogReg.predict(new_input)
recommended_crop = predicted_values[0]

print("Predicted Crop (Logistic Regression):", recommended_crop)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(7,)),  # 7 input features
    tf.keras.layers.Dense(1, activation='sigmoid')  # 1 output neuron with sigmoid activation
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Encode the labels using LabelEncoder
label_encoder = LabelEncoder()
Y_train_encoded = label_encoder.fit_transform(Y_train)

model.fit(X_train, Y_train_encoded, epochs=50, verbose=0)

predicted_values_tf_encoded = model.predict(new_input)
predicted_class = int(round(predicted_values_tf_encoded[0][0]))  # Round and convert to integer
reverse_encoded_label = label_encoder.inverse_transform([predicted_class])

print("Neural Network predicted crop (TensorFlow):", reverse_encoded_label[0])
