from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf


app = Flask(__name__)
CORS(app)  # Initialize CORS with your Flask app
# Your code for data loading, model training, and prediction here
# ...

# @app.route('/predict', methods=['POST'])
# def predict():
#     if request.method == 'POST':
#         # Extract input data from the POST request
#         input_data = request.json  # Assuming JSON data is sent in the request
#         PATH = 'Crop_recommendation.csv'
#         data = pd.read_csv(PATH)
#         stringData = (str(data).split("\n"))
#         arrayData = (stringData[len(stringData) - 3].split()[4:-1])
#         numData = [float(item) for item in arrayData]
#         # for i in range(len(stringData) - 1, 0, -1):
#         #     if stringData[i] != "" and stringData[i] != " ":
#         #         print(stringData[i])
#         #         print(i)
#
#         input_data = data[['temperature', 'humidity', 'ph', 'rainfall']]
#         output_crop = data['label']
#
#         X_train, X_test, Y_train, Y_test = train_test_split(input_data, output_crop, test_size=0.2, random_state=42)
#
#         LogReg = LogisticRegression(random_state=42).fit(X_train, Y_train)
#
#         new_input = np.array([fdata])
#         predicted_values = LogReg.predict(new_input)
#         recommended_crop = predicted_values[0]
#
#         print("Predicted Crop (Logistic Regression):", recommended_crop)
#
#         model = tf.keras.Sequential([
#             tf.keras.layers.Input(shape=(7,)),  # 7 input features
#             tf.keras.layers.Dense(1, activation='sigmoid')  # 1 output neuron with sigmoid activation
#         ])
#
#         model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#
#         # Encode the labels using LabelEncoder
#         label_encoder = LabelEncoder()
#         Y_train_encoded = label_encoder.fit_transform(Y_train)
#
#         model.fit(X_train, Y_train_encoded, epochs=50, verbose=0)
#
#         predicted_values_tf_encoded = model.predict(new_input)
#         predicted_class = int(round(predicted_values_tf_encoded[0][0]))  # Round and convert to integer
#         reverse_encoded_label = label_encoder.inverse_transform([predicted_class])
#
#         print("Neural Network predicted crop (TensorFlow):", reverse_encoded_label[0])
#         # Perform predictions using your Logistic Regression and Neural Network models
#         # ...
#
#         # Create a response JSON object
#         response = {
#             "LogisticRegression": recommended_crop,  # Replace with the actual result
#             "NeuralNetwork": reverse_encoded_label[0]  # Replace with the actual result
#         }
#
#         return jsonify(response)
#

@app.route('/get_recommendation', methods=['GET'])
def get_recommendation():
    if request.method == 'GET':
        # You can include logic here to obtain the recommended crop using your models
        # For example, you can use predefined input data or request parameters
        # Replace the following line with your logic
        # recommended_crop = "Replace this with the recommended crop based on your logic"
        PATH = 'Crop_recommendation.csv'
        data = pd.read_csv(PATH)

        input_data = data[['temperature', 'humidity', 'ph', 'rainfall']]
        output_crop = data['label']

        X_train, X_test, Y_train, Y_test = train_test_split(input_data, output_crop, test_size=0.2, random_state=42)

        LogReg = LogisticRegression(random_state=42).fit(X_train, Y_train)
        read = pd.read_csv("Data.csv")
        readData = (str(read).split("\n"))
        c_data = [element for element in (readData[len(readData) - 1].split(" ")) if element != '']
        print(c_data)
        fdata = [float(e) for e in c_data[-4:]]
        print(fdata)
        new_input = np.array([fdata])
        predicted_values = LogReg.predict(new_input)
        recommended_crop = predicted_values[0]

        print("Predicted Crop (Logistic Regression):", recommended_crop)

        # model = tf.keras.Sequential([
        #     tf.keras.layers.Input(shape=(7,)),  # 7 input features
        #     tf.keras.layers.Dense(1, activation='sigmoid')  # 1 output neuron with sigmoid activation
        # ])
        #
        # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        #
        # # Encode the labels using LabelEncoder
        # label_encoder = LabelEncoder()
        # Y_train_encoded = label_encoder.fit_transform(Y_train)
        #
        # model.fit(X_train, Y_train_encoded, epochs=50, verbose=0)
        #
        # predicted_values_tf_encoded = model.predict(new_input)
        # predicted_class = int(round(predicted_values_tf_encoded[0][0]))  # Round and convert to integer
        # reverse_encoded_label = label_encoder.inverse_transform([predicted_class])
        #
        # print("Neural Network predicted crop (TensorFlow):", reverse_encoded_label[0])
        # # Create a response JSON object
        response = {
            "RecommendedCrop": recommended_crop,
            "Temperature": fdata[0],
            "Humidity": fdata[1],
            "Light": fdata[2],
            "Rainfall": fdata[3]
        }

        return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True)
