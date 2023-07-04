import os
os.getcwd()
# Get the root directory of the current drive
root_dir = os.path.abspath(os.sep)
root_dir
# Importing required functions
from flask import Flask, render_template
import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib

# Flask constructor
app = Flask(__name__)

# Generate a scatter plot and returns the figure
def get_plot():

    data = {
        'a': np.arange(50),
        'c': np.random.randint(0, 50, 50),
        'd': np.random.randn(50)
    }

    data['b'] = data['a'] + 10 * np.random.randn(50)
    data['d'] = np.abs(data['d']) * 100

    plt.scatter('a', 'b', c='c', s='d', data=data)
    plt.xlabel('X label')
    plt.ylabel('Y label')

    return plt

# Root URL
@app.route('/')
def single_converter():
    # Get the matplotlib plot
    plot = get_plot()

    # Save the figure in the static directory
    plot.savefig(os.path.join('static', 'images', 'plot.png'))

    # Close the figure to avoid overwriting
    plot.close()
    return render_template('matplotlib-plot1.html')

# Main Driver Function
if __name__ == '__main__':
    # Run the application on the local development server
    app.run(debug=True)

