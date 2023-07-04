FROM tensorflow/serving

ENV MODEL_NAME=basic_model
RUN mkdir -p /models/$MODEL_NAME \ 
    apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*
COPY ./basic_model /models/$MODEL_NAME
