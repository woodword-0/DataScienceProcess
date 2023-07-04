import streamlit as st
import time
from PIL import Image
import joblib
from sklearn.feature_extraction.text import CountVectorizer 
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import numpy as np
###################################################################################
###################################################################################
# Import Dataframe
df = pd.read_excel('Data.xlsx')
###################################################################################
###################################################################################
# Restricted Dataframe
df1 = df[['DateCreated','ItemName','CustomerId']].copy()
df1 = df1.loc[df1.CustomerId == -1]
df1.dropna(inplace = True)
###################################################################################
###################################################################################
# List of all oredered items per customer
corpus = [df1.ItemName.iloc[i] for i in range(len(df1))]
###################################################################################
###################################################################################
# # Term frequency
df1 = df1.loc[df1.CustomerId == -1]
###################################################################################
###################################################################################
# Count vectorizer
# count_vectorizer = CountVectorizer(token_pattern='(?u)\\b\\w+(?:\\s\\w+)*\\b',stop_words='english', ngram_range=(0,2))
count_vectorizer = CountVectorizer(stop_words='english', ngram_range=(0,2))

# corpus2 = count_vectorizer.fit_transform(df1.ItemName)
corpus2 = count_vectorizer.fit_transform(df1.ItemName)

###################################################################################
###################################################################################
# Cosine Similarity
df2 = pd.DataFrame(cosine_similarity(corpus2, dense_output=True))
###################################################################################
###################################################################################
# Processing cosine similarity

# Step 1: Put all cosine values into a list
t = []
for j,k in enumerate(df2.values):
    for n in range(len(k)):
        t.append([j,n,k[n]])
        
# Step 2: Drop any diagonal values
qq = []
for i in range(len(t)):
    if t[i][0]==t[i][1]:
        qq.append([t[i][0],t[i][1],0])
    else:
        qq.append(t[i])
        
# Step 3: Create new dataframe
from collections import defaultdict
u = defaultdict(list)
for i in range(len(qq)):
    u[qq[i][0]].append(qq[i][2])
    
updated_df = pd.DataFrame(u)

# Step 4: Find maximally paired indices
position_maxVal=[]
for i in range(len(updated_df)):
    position_maxVal.append(np.argmax(updated_df[i]))
    
        
# Step 5: Dataframe of similar items
sent_comp = []
for j in position_maxVal:
    sent_comp.append(corpus[j])
sent_comp
## Items based on highest similarity per row as df
similar_items = pd.DataFrame(sent_comp,columns=['SimilarItems'])
## Cosine similarity values
similarity_value = pd.DataFrame(round(updated_df.max(axis=1),4),columns=['SimilarityValue'])
## Item names
p_itm = pd.DataFrame(corpus,columns=["Items"])
## join dataframes 
cos_sim_df = pd.concat([p_itm,similar_items,similarity_value],axis=1)
## Drop redundancies
cos_sim_df = cos_sim_df.loc[cos_sim_df['SimilarityValue'] < 1]
cos_sim_df.shape
cos_sim_df

cosine_sim = cosine_similarity(tfidf,tfidf)
cosine_sim.shape
cosine_sim_df = pd.DataFrame(cosine_sim, columns=df['ItemName'], index=df['ItemName'])
cos_sim_df.columns


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
