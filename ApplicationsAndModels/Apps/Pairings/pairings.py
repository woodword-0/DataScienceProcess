import streamlit as st
import time
from PIL import Image
import joblib
from sklearn.feature_extraction.text import CountVectorizer 
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
# Read the Excel file
df = pd.read_excel('Data.xlsx')
demail = df.Email.dropna()
df1 = df.copy()
df1 = df1[['DateCreated','ItemName','Email']]
df1.shape
df1.dropna(inplace = True)
df1.isna().sum()
df.DateCreated

cv = CountVectorizer(token_pattern='(?u)\\b\\w+(?:\\s\\w+)*\\b')

item_matrix = cv.fit_transform(df['ItemName']).transpose().dot(cv.fit_transform(df['ItemName']))
# Add title to page
# cv.vocabulary_
st.title("Recommender - PringleAPI")
# Create a DataFrame from the vocabulary
vocabulary_df = pd.DataFrame(list(cv.vocabulary_.items()), columns=['word', 'index'])
input_word = st.text_input('Enter a dish name:')
st.dataframe(vocabulary_df)
# Create a submit button
if st.button('Recommend'):
    # Transform the input word into a TF-IDF score vector
    input_index = cv.vocabulary_[input_word.lower()]
    item_vector = item_matrix[input_index]
    # Calculate the cosine similarity between the input vector and each item vector
    similarity_scores = cosine_similarity(item_vector, item_matrix)

    # Get the indices of the top 3 most similar items
    most_similar_indices = similarity_scores.argsort()[0][-3:][::-1]

    # Get the names of the most similar items
    most_similar_items = df.iloc[most_similar_indices]['ItemName'].values

    # Display the result
    st.write(f'The best paired dishes with  "{input_word}" are:')
    for item in most_similar_items:
        st.write(item)


## Pairings Recommender

df = df[['ItemName','CustomerId']]
df.dropna(inplace=True)
# Create a co-occurrence matrix of the items
# cv = CountVectorizer(token_pattern='(?u)\\b\\w+\\b')
cv = CountVectorizer(token_pattern='(?u)\\b\\w+(?:\\s\\w+)*\\b')

item_matrix = cv.fit_transform(df['ItemName']).transpose().dot(cv.fit_transform(df['ItemName']))
# Add title to page
# cv.vocabulary_
st.title("Recommender - PringleAPI")
# Create a DataFrame from the vocabulary
vocabulary_df = pd.DataFrame(list(cv.vocabulary_.items()), columns=['word', 'index'])
input_word = st.text_input('Enter a dish name:')
st.dataframe(vocabulary_df)
# Create a submit button
if st.button('Recommend'):
    # Transform the input word into a TF-IDF score vector
    input_index = cv.vocabulary_[input_word.lower()]
    item_vector = item_matrix[input_index]
    # Calculate the cosine similarity between the input vector and each item vector
    similarity_scores = cosine_similarity(item_vector, item_matrix)

    # Get the indices of the top 3 most similar items
    most_similar_indices = similarity_scores.argsort()[0][-3:][::-1]

    # Get the names of the most similar items
    most_similar_items = df.iloc[most_similar_indices]['ItemName'].values

    # Display the result
    st.write(f'The best paired dishes with  "{input_word}" are:')
    for item in most_similar_items:
        st.write(item)
