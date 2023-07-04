
import streamlit as st
# import time
# from PIL import Image
# import joblib
# from sklearn.feature_extraction.text import CountVectorizer 
import pandas as pd



# Read the Excel file
df = pd.read_excel('Data.xlsx')
# Count the number of occurrences of each dish and sort in descending order
top_dishes = df['ItemName'].value_counts().sort_values(ascending=False)

# Get the top 5 most popular dishes as a list of tuples
top_five = list(top_dishes.head(5).items())
top_five = [top_five[i][0] for i in range(len(top_five))]
# Render a template with the top 5 dishes    
st.write(f'The top 5 most commonly ordered dishes at Aappakadai Indian Chettinad - Santa Clara, CA are:')
for item in top_five:
    st.write(item)
st.dataframe(top_dishes)