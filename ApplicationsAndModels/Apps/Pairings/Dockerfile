FROM python:3.8-slim-buster

# Set the working directory
WORKDIR /app

# Copy the requirements file and install the necessary packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app files to the container
COPY . .

# Set the environment variable for Streamlit
ENV STREAMLIT_SERVER_PORT 8501

# Expose the port for the Streamlit app
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "pairings.py", "--server.enableCORS", "false"]
