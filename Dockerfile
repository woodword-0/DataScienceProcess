FROM python:3.9

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "flasktest.py","flask", "run", "--host=0.0.0.0", "--port=5000"]


# http://20.122.215.207/:5000