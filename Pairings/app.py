import streamlit as st
import time
from PIL import Image
import joblib
from sklearn.feature_extraction.text import CountVectorizer 
# gender_cv = CountVectorizer()
# # Load the model
# gender_nv_model = open("model/naivemodel.pkl", "rb")
# gender_clf = joblib.load(gender_nv_model)
# gender_nv_model.close()
# # Prediction Logic
# def predict_gender(data):
#   vect = gender_cv.transform(data).toarray()
#   result = gender_clf.predict(vect)
#   return result
# import streamlit as st
# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
path = 'C:/CodeTest1/Venv1/SalesFeb2023/pringledatascience/pringledatascience/Recommender/Data.xlsx'

# # Read the Excel file and create the TF-IDF vectorizer
# df = pd.read_excel(path)
# df.ItemName.shape
# df = df[['ItemName','CustomerId']]
# df = df.dropna()
# vectorizer = TfidfVectorizer()
# vectorizer.fit(df['ItemName'])
# vectorizer

# Define the Streamlit app
# def app():
    # Create the input field for the item name
# input_word = st.text_input('Enter a dish name:')
# input_word = 'IDLY'
# Create a submit button
# if st.button('Recommend'):
    # Transform the input word into a TF-IDF score vector
# input_vector = vectorizer.transform([input_word])
# input_vector
# # Calculate the cosine similarity between the input vector and each item vector
# similarity_scores = cosine_similarity(input_vector, vectorizer.transform(df['ItemName']))
# [x for x in similarity_scores]
# # Get the indices of the top 3 most similar items
# most_similar_indices = similarity_scores.argsort()[0][-3:][::-1]
# most_similar_indices
# # Get the names of the most similar items
# most_similar_items = df.iloc[most_similar_indices]['ItemName'].values
# most_similar_items
#     # Display the result
#     st.write(f'The top 3 most similar dishes to "{input_word}" are:')
#     for item in most_similar_items:
#         st.write(item)




# import streamlit as st

# def return_NER(value):
#     doc = nlp(value)
#     return [(X.text, X.label_) for X in doc.ents]

# Add title on the page
# st.title("Spacy - Named Entity Recognition")









# Ask user for input text
# input_sent = st.text_input("Input Sentence", "Your input sentence goes here")
# print(cv.vocabulary_)

# Display named entities
# for res in return_NER(input_sent):
#     st.write(res[0], "-->", res[1])




import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Read the Excel file
df = pd.read_excel(path)
df = df[['ItemName','CustomerId']]
df.dropna(inplace=True)
input_word = 'idly'
# Create a co-occurrence matrix of the items
cv = CountVectorizer(token_pattern='(?u)\\b\\w+\\b')
item_matrix = cv.fit_transform(df['ItemName']).transpose().dot(cv.fit_transform(df['ItemName']))
input_index = cv.vocabulary_[input_word]
item_vector = item_matrix[input_index]
# Calculate the cosine similarity between the input vector and each item vector
similarity_scores = cosine_similarity(item_vector, item_matrix)

# Get the indices of the top 3 most similar items
most_similar_indices = similarity_scores.argsort()[0][-3:][::-1]

# Get the names of the most similar items
most_similar_items = df.iloc[most_similar_indices]['ItemName'].values
# Get the input word from the user
# input_word = input('Enter a word: ')
# df.ItemName
# Calculate the cosine similarity between the input item and each item
# input_index = cv.vocabulary_[input_word]
# item_vector = item_matrix[input_index]
# similarity_scores = cosine_similarity(item_vector, item_matrix)

# Get the top three most frequently ordered items
# most_similar_indexes = similarity_scores.argsort()[0][::-1][1:4]
# most_similar_items = [cv.get_feature_names()[i] for i in most_similar_indexes]
# def app():
    # Create the input field for the item name
    # Add title on the page
st.title("Recommender - PringleAPI")


input_word = st.text_input('Enter a dish name:')

# Create a submit button
if st.button('Recommend'):
    # Transform the input word into a TF-IDF score vector
    input_index = cv.vocabulary_[input_word]
    item_vector = item_matrix[input_index]
    # Calculate the cosine similarity between the input vector and each item vector
    similarity_scores = cosine_similarity(item_vector, item_matrix)

    # Get the indices of the top 3 most similar items
    most_similar_indices = similarity_scores.argsort()[0][-3:][::-1]

    # Get the names of the most similar items
    most_similar_items = df.iloc[most_similar_indices]['ItemName'].values

    # Display the result
    st.write(f'The top 3 most commonly ordered dishes with  "{input_word}" are:')
    for item in most_similar_items:
        st.write(item)
# Latex
# st.latex(r'''
#     a + ar + a r^2 + a r^3 + \cdots + a r^{n-1} =
#     \sum_{k=0}^{n-1} ar^k =
#     a \left(\frac{1-r^{n}}{1-r}\right)
#     ''')

# import streamlit as st
# import pandas as pd
# import numpy as np

# chart_data = pd.DataFrame(
#     np.random.randn(20, 3),
#     columns=["a", "b", "c"])

st.bar_chart(df.ItemName)





# Print the result
# print(f'The top three most frequently ordered together items with "{input_word}" are: {most_similar_items}')

# Define the Streamlit app
# def app():
    # Create the input field for the item name
# input_word = 'IDLY'
# Create a submit button
# if st.button('Recommend'):
    # Transform the input word into a TF-IDF score vector
# input_vector = vectorizer.transform([input_word])
# input_vector
# # Calculate the cosine similarity between the input vector and each item vector
# similarity_scores = cosine_similarity(input_vector, vectorizer.transform(df['ItemName']))
# [x for x in similarity_scores]
# # Get the indices of the top 3 most similar items
# most_similar_indices = similarity_scores.argsort()[0][-3:][::-1]
# most_similar_indices
# # Get the names of the most similar items
# most_similar_items = df.iloc[most_similar_indices]['ItemName'].values
# most_similar_items
    # Display the result
# st.write(f'The top 3 most similar dishes to "{input_word}" are:')
# for item in most_similar_items:
#     st.write(item)




# import streamlit as st

# def return_NER(value):
#     doc = nlp(value)
#     return [(X.text, X.label_) for X in doc.ents]

# Add title on the page
# st.title("Spacy - Named Entity Recognition")









# Ask user for input text
# input_sent = st.text_input("Input Sentence", "Your input sentence goes here")

# Display named entities
# for res in return_NER(input_sent):
#     st.write(res[0], "-->", res[1])































# Styling the app
# def load_css(file_name):
#     with open(file_name) as f:
#         st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)

# def load_icon(icon_name):
#     st.markdown('<i class="material-icons">{}</i>'.format(icon_name), unsafe_allow_html=True)
    
# # Adding Images
# def load_images(file_name):
#   img = Image.open(file_name)
#   return st.image(img,width=300)
    
# # Designing the User Interface
# def main():
#   """Gender Classifier App
#     With Streamlit

#   """

#   st.title("Gender Classifier")
#   html_temp = """
#   <div style="background-color:blue;padding:10px">
#   <h2 style="color:grey;text-align:center;">Streamlit App </h2>
#   </div>

#   """
#   st.markdown(html_temp,unsafe_allow_html=True)
#   load_css('icon.css')
#   load_icon('people')

#   name = st.text_input("Enter Name","Pleas Type Here")
#   if st.button("Predict"):
#     result = predict_gender([name])
#     if result[0] == 0:
#       prediction = 'Female'
#       img = 'female.png'
#     else:
#       result[0] == 1
#       prediction = 'Male'
#       img = 'male.png'

#     st.success('Name: {} was classified as {}'.format(name.title(),prediction))
#     load_images(img)