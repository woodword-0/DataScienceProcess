import tensorflow as tf
path = "C:/Users/TurnerJ/OneDrive/Desktop/Docker0/"
# Define the model architecture
model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(units=1, input_shape=[1])
])

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(1e-1), loss='mean_squared_error')
# Save the model in the SavedModel format
# tf.saved_model.save(model, path + 'basic_model')
'saved_model_cli show --all --'
model.summary()
import requests
import json

# set the url for the TensorFlow Serving container
url = 'http://localhost:8501/v1/models/model1/metadata'

# send a GET request to the url to get the model metadata
response = requests.get(url)

# parse the JSON response
model_metadata = json.loads(response.content.decode('utf-8'))
model_metadata.keys()
# get the model signature
signature = model_metadata['metadata']['signature_def']['signature_def']

# print the model signature
print('Model signature: ', signature)
model_metadata['model_spec']
model_metadata['metadata']

# get the model summary
summary = model_metadata['metadata']['model_summary']

# print the model summary
print('Model summary: ', summary)


import requests

url = "http://localhost:8501/v1/models/model1/metadata"
response = requests.get(url)
data = response.json()

print("Model signature:")
print(data['metadata']['signature_def']['serving_default'])

print("\nModel input shape:")
print(data['metadata']['model_spec']['signature_name']['serving_default']['inputs'])

print("\nModel output shape:")
print(data['metadata']['model_spec']['signature_name']['serving_default']['outputs'])



import grpc
import numpy as np
import tensorflow as tf
from tensorflow_serving.apis import prediction_service_pb2_grpc, predict_pb2

# Define the input data
input_data = np.random.rand(1, 28, 28, 1).astype(np.float32)

# Define the address and port of the serving model
server_address = 'localhost:8501'

# Create a gRPC channel
channel = grpc.insecure_channel(server_address)

# Create a stub for making prediction requests
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

# Define the name of the model to use
model_name = 'model1'

# Define the version of the model to use (use -1 for latest version)
model_version = -1

# Define the name of the input tensor (use the name printed in the signature)
input_tensor = 'conv2d_input'

# Create a request message with the input data
request = predict_pb2.PredictRequest()
request.model_spec.name = model_name
request.model_spec.version.value = model_version
request.inputs[input_tensor].CopyFrom(tf.make_tensor_proto(input_data))
request
# Send the prediction request and get the response
response = stub.Predict(request)

# Get the output tensor names (use the names printed in the signature)
output_tensor_names = ['dense_1/Softmax']

# Get the output data from the response message
output_data = tf.make_ndarray(response.outputs[output_tensor_names[0]])

# Print the predicted class
predicted_class = np.argmax(output_data)
print('Predicted class:', predicted_class)


import grpc
import numpy as np
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

# Create a gRPC channel and stub to connect to the server
channel = grpc.insecure_channel('localhost:8500')
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

# Set up the request object with the model name and signature name
request = predict_pb2.PredictRequest()
request.model_spec.name = 'model1'
request.model_spec.signature_name = 'serving_default'

# Set up the input tensor with some example data
input_data = np.array([[1.0], [2.0], [3.0], [4.0]], dtype=np.float32)
request.inputs['dense_input'].CopyFrom(tf.make_tensor_proto(input_data, shape=input_data.shape))

# Make the prediction and print the output
response = stub.Predict(request, timeout=10.0)
type(response)
output_data = tf.make_ndarray(response.outputs['dense'])
print(output_data)


import requests

# Make a prediction request to the server
data = {
    "instances": [
        {"dense_input": [1.0]},
        {"dense_input": [2.0]},
        {"dense_input": [3.0]}
    ]
}
response = requests.post('http://localhost:8501/v1/models/model1:predict', json=data)

# Get the JSON data from the response
json_data = response.json()

# Print the predictions
print(json_data)

