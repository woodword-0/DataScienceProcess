from flask import Flask, render_template, request, jsonify
import json
import requests

app = Flask(__name__)

# Define the URL for the predict request
PREDICT_URL = "http://localhost:8501/v1/models/basic_model:predict"

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Get the input data from the form
        input_data = {
            "instances": [
                {"dense_input": [float(request.form['input1'])]},
                {"dense_input": [float(request.form['input2'])]},
                {"dense_input": [float(request.form['input3'])]}
            ]
        }

        # Convert input data to JSON
        input_data_json = json.dumps(input_data)

        # Define headers
        headers = {"content-type": "application/json"}

        # Make the request to the TensorFlow serving container
        response = requests.post(PREDICT_URL, data=input_data_json, headers=headers)

        # Parse the response JSON
        output_data = json.loads(response.content.decode('utf-8'))

        # Format the output data for display in the browser
        output1 = output_data['predictions'][0][0]
        output2 = output_data['predictions'][1][0]
        output3 = output_data['predictions'][2][0]

        return render_template('index.html', output1=output1, output2=output2, output3=output3)
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
