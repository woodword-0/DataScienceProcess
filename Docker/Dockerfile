# Use the TensorFlow Serving image as the base image
FROM tensorflow/serving

# Create a directory to store the model
RUN mkdir -p /models/my_model

# Copy the saved model to the container
COPY /mnt/c/CodeTest1/Venv1/SalesFeb2023/pringledatascience/pringledatascience/Docker/linear_model /models/my_model

# Expose the default serving port for TensorFlow Serving
EXPOSE 8501

# Start TensorFlow Serving when the container starts
CMD ["tensorflow_model_server", "--port=8500", "--rest_api_port=8501", "--model_name=linear_model", "--model_base_path=/models/my_model"]
