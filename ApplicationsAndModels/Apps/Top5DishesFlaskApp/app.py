from flask import Flask, render_template
import pandas as pd

app = Flask(__name__)

@app.route('/')
def top_five():
    path = 'C:/CodeTest1/Venv1/SalesFeb2023/pringledatascience/pringledatascience/Recommender/Data.xlsx'

    # Read the Excel file
    df = pd.read_excel(path)

    # Count the number of occurrences of each dish and sort in descending order
    top_dishes = df['ItemName'].value_counts().sort_values(ascending=False)

    # Get the top 5 most popular dishes as a list of tuples
    top_five = list(top_dishes.head(5).items())
    # Render a template with the top 5 dishes
    return render_template('index.html', top_five=top_five)

if __name__ == '__main__':
    app.run(port = 5001,debug=True)
