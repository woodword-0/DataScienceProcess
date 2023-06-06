import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.feature_extraction import DictVectorizer 
from sklearn.model_selection import train_test_split 
df = pd.read_csv('dataset.csv')
df.size
df.dtypes
df.isnull().isnull().sum()

df[df.sex == 'F'].size
df[df.sex == 'M'].size
df_names = df

# Replace genders with 0 or 1 
df_names.sex.replace({'F':0,'M':1},inplace=True)
Xfeatures =df_names['name']

# Feature Extraction
cv = CountVectorizer()
X = cv.fit_transform(Xfeatures)
#Processing the model
X
y = df_names.sex
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
# Create Naive Bayes Classifier
from sklearn.naive_bayes import MultinomialNB 
clf = MultinomialNB()
clf.fit(X_train,y_train)
clf.score(X_test,y_test)

#Prediction function
def predict(a):
    test_name = [a]
    vector = cv.transform(test_name).toarray()
    if clf.predict(vector) == 0:
        print("Female")
    else:
        print("Male")
        
# Pickle the model
import joblib

# Pickle the model
joblib.dump(clf, 'naivemodel.pkl')
