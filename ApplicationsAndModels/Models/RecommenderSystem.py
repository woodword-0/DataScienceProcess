path = 'C:/CodeTest1/Venv1/SalesFeb2023/pringledatascience/pringledatascience/Recommender/Data.xlsx'
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Read the Excel file
df = pd.read_excel(path)
df = df[['ItemName','CustomerId']]
df.shape
df = df.dropna()
df.isna().sum()
# df.CustomerId

df.CustomerId




# df.nunique()
# df.TotalCost.isna().sum()
# df.CartID.nunique()
# # Note repeats in dataframe
# df[['CustomerId','CartID','DateCreated']]
# # find all duplicate rows
# duplicates = df[df.duplicated()]
# co = df.iloc[[33719,33720,33721,33722,33723]].columns 
# df.iloc[[33719,33720,33721,33722,33723]][co[:10]]
# df.iloc[[33719,33720,33721,33722,33723]][co[10:20]]
# df.iloc[[33719,33720,33721,33722,33723]][co[20:]]
# df.CartItemID.nunique()
# len(co)
# print(duplicates)
# df.drop('CustomerCategory',axis=1).duplicated().sum()
# # Display the contents of the file
# print(df.head())
# df.isna().sum()
# df.DateCreated.nunique()
# cols = ['CartID', 'DateCreated','CustomerId', 'TotalCost', 'CartItemID', 'CartID.1', 'MenuItemID', 'Quantity',
#        'UnitCost', 'MenuItemID.1', 'MenuCategoryID', 'ItemName',
#        'ItemDescription', 'ItemPrice']
# df[cols].loc[df[cols]['CustomerId'] == -1].head(3)
# df[cols].isna().sum()
# import pandas as pd
# # All Data for one customer
# cust0 = df.loc[df.CustomerId == 0]
# cust0.isna().sum()
# #####################################################################################
# #####################################################################################
# cust0_orders = df.loc[df.CustomerId == 0][['ItemName','MenuItemID']]
# cust0.groupby('DateCreated')['ItemName'].agg(list)
# cust0.groupby(['DateCreated', 'ItemName', 'ItemDescription'])['Quantity'].sum().reset_index()
# import pandas as pd
# import matplotlib.pyplot as plt
# #####################################################################################
# #####################################################################################
# MenuItems  = cust0_orders.groupby(cust0_orders['ItemName'])

# i = 1
# for item, id in MenuItems:
#     print(str(i)+' '+item + ' : '+ str(len(id)) )
#     i = i+1

# plt.figure(figsize=(10,10))
# sns.countplot(y='ItemName', data = cust0_orders )
# plt.show()







# # Create a dictionary mapping MenuItemId to ItemName
# item_dict = dict(zip(cust0.MenuItemID, cust0.ItemName))

# # Group the dataframe by MenuItemID and count the number of occurrences
# item_counts = cust0.groupby('MenuItemID').size()

# # Filter out the bins with zero counts
# non_zero_counts = item_counts[item_counts != 0]
# non_zero_indices = non_zero_counts.index

# # Create a list of labels for the bins using the item_dict
# labels = [item_dict.get(mid) for mid in non_zero_indices]

# # Create a histogram using matplotlib
# fig, ax = plt.subplots(figsize=(15, 7))
# ax.hist(non_zero_indices, bins=len(non_zero_counts), weights=non_zero_counts.values, edgecolor='black')
# ax.set_xticks(non_zero_indices)
# ax.set_xticklabels(labels, rotation=45)
# ax.set_xlabel('Item Name')
# ax.set_ylabel('Count')
# ax.set_title('Histogram of Items Ordered')
# plt.show()



# df.CustomerId.nunique()
# #####################################################################################
# #####################################################################################

# # Create a dictionary mapping MenuItemId to ItemName
# item_dict = dict(zip(cust0.MenuItemID, cust0.ItemName))
# cust0.ItemName.shape
# # Group the dataframe by MenuItemID and count the number of occurrences
# item_counts = cust0.groupby('ItemName').size()

# item_counts.hist()
# plt.show()
# # Plot the data as a histogram
# item_counts.hist(figsize=(10, 5), edgecolor='black')
# item_counts.max()
# # Set the axis labels and title
# plt.xlabel('Count')
# plt.ylabel('Item Name')
# plt.title('Histogram of Items Ordered')

# # Show the plot
# plt.show()
# df['Token.1'].nunique()
# df.DateCreated.nunique()
# ################################################################################################################################################
# ################################################################################################################################################
# cols = ['CustomerId','CartItemID','CartID','ItemName','TotalCost']
# df[cols].nunique()
# df[cols].head(3)
# orders = df.groupby()
# ################################################################################################################################################
# ################################################################################################################################################
# # Frequency DataFrame
# # Total Orders
# total_orders = df.CartID.nunique()
# total_orders
# # Total Sales
# # Remove duplicates based on CartID
# unique_df = df.drop_duplicates(subset=['CartID'])
# sales = unique_df.TotalCost.sum()
# # Group by CartID and calculate the sum of TotalCost
# cart_totals = unique_df.groupby('CartID')['TotalCost'].sum()
# cart_totals
# # Create a DataFrame from the cart_totals Series
# cart_totals_df = pd.DataFrame({'CartID': cart_totals.index, 'TotalSales': cart_totals.values})

# total_sales = df.groupby('CartID')['TotalCost'].sum()
# total_sales
# # List of orders placed by each customer
# # Group the dataframe by 'CustomerId' and calculate the sum of 'TotalCost' for each individual customer
# cart_totals = df.groupby('CustomerId')['TotalCost'].sum()
# cart_totals = cart_totals.reset_index()

# cart_totals
# # Group the DataFrame by CustomerId and total number of orders per customer using CartId
# # Each Row is a unique customer Id an the total number of orders ordered by said customer
# customer_orders = df.groupby('CustomerId')['CartID'].count()
# customer_orders = customer_orders.reset_index()
# customer_orders
# # Combine DataFrames
# customer_orders = customer_orders.merge(cart_totals, on='CustomerId').drop_duplicates()
# customer_orders.isna().sum()

# # Add the order frequency per customer
# customer_orders['OrderFreq'] = (customer_orders['CartID']/total_orders)
# customer_orders.OrderFreq.sum()
# customer_orders.columns
# # Add the order value
# customer_orders['Value'] = (customer_orders['TotalCost']/sales)


# ################################################################################################################################################
# ################################################################################################################################################
# Basic Recommender
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
df.ItemName.shape
# Create a TF-IDF vectorizer and fit it to the item names
vectorizer = TfidfVectorizer()
vectorizer.fit(df)
df.ItemName.isna().sum()
vectorizer.fit(df['ItemName'])

# Get the input word from the user
input_word = input('Enter a word: ')
input_word
# Create a matrix of TF-IDF scores for each item
tfidf_matrix = vectorizer.transform(df)

df1 = df.ItemName.astype('string')
df1
# Transform the input word into a TF-IDF score vector
input_vector = vectorizer.transform([input_word])
input_vector = vectorizer.transform(['IDLY'])
# Calculate the cosine similarity between the input vector and each item vector
similarity_scores = cosine_similarity(input_vector, vectorizer.transform(df))
similarity_scores
# Find the index of the item with the highest cosine similarity score
most_similar_index = similarity_scores.argmax()
most_similar_index
# Get the name of the most similar item
most_similar_item = df.iloc[most_similar_index]

# Print the result
print(f'The most similar dish to "{input_word}" is "{most_similar_item}"')



from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
import pandas as pd

# Load the data into a DataFrame
df = pd.read_csv('data.csv')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# Create a TfidfVectorizer object and fit it to the training data
vectorizer = TfidfVectorizer()
X_train_transformed = vectorizer.fit_transform(X_train)

# Train a machine learning model on the transformed data
clf = LinearSVC()
clf.fit(X_train_transformed, y_train)

# Use the trained model to predict labels for the test data
X_test_transformed = vectorizer.transform(X_test)
y_pred = clf.predict(X_test_transformed)

# Create a confusion matrix using the true labels and predicted labels
conf_matrix = confusion_matrix(y_test, y_pred)

# Print the confusion matrix
print(conf_matrix)










# from sklearn.feature_extraction.text import TfidfVectorizer

# # Define a list of item names
# item_names = ['CHICKEN 65', 'CHICKEN BIRYANI', 'VEGETABLE SAMOSA', 'PANEER TIKKA', 'VEGETABLE PAKORA']

# # Create a TF-IDF vectorizer and fit it to the item names
# vectorizer = TfidfVectorizer()
# vectorizer.fit(item_names)

# # Transform a new set of item names using the fitted vectorizer
# new_item_names = ['CHICKEN CURRY', 'PANEER MASALA']
# tfidf_matrix = vectorizer.transform(new_item_names)

# # Print the transformed matrix
# print(tfidf_matrix.toarray())




# idx = df.index[df['ItemName'] == item_name][0]


tfidf_matrix.shape

tfidf_matrix

df.shape

# Calculate the cosine similarity between items
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
# from sklearn.feature_extraction.text import TfidfVectorizer
cosine_sim.shape
# # Define a list of item names
# item_names = ['CHICKEN 65', 'CHICKEN BIRYANI', 'VEGETABLE SAMOSA', 'PANEER TIKKA', 'VEGETABLE PAKORA']

# # Create a TF-IDF vectorizer and fit it to the item names
# vectorizer = TfidfVectorizer()
# vectorizer.fit(item_names)

# # Transform a new set of item names using the fitted vectorizer
# new_item_names = ['CHICKEN CURRY', 'PANEER MASALA']
# tfidf_matrix = vectorizer.transform(new_item_names)
df = df.astype('str')
df.isna().sum()
df.dtypes
type(df)
df
idx = df.index[df = 'IDLY']
idx
[df.iloc[i] for i in range(len(df))]
# # Print the transformed matrix
# print(tfidf_matrix.toarray())
(df == 'IDLY').sum()
# Define a function to get recommendations for a given item
def get_recommendations(item_name, cosine_sim, df):
    # Get the index of the item in the dataframe
    idx = df.index[df['ItemName'] == item_name][0]

    # Get the cosine similarity scores for all items
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the items by their similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the top 5 most similar items (excluding the item itself)
    top_items = [i[0] for i in sim_scores[1:6]]

    # Return the names of the top items
    return df.loc[top_items]['ItemName']

# # Example usage: get recommendations for the item "Cheeseburger"
# get_recommendations('IDLY', cosine_sim, df)
################################################################################################################################################
################################################################################################################################################


# customer_orders = df.groupby('CustomerId')['CartItemID'].agg(list)
# customer_orders
# df.nunique()
# df.CartItemID.nunique()

# customer_orders.shape
# # CustomerId - unique id for each customer
# # CartID - unique for each order
# # CartItemID - unique for each item in each order
# # TotalCost - Total cost of order


# # Reset the index to convert CustomerId into a column
# customer_orders = customer_orders.reset_index()
# df.CustomerId.nunique()
# df.loc[df.CustomerId == 0].shape
# # customer_orders['CartTotal'] = cart_totals
# customer_orders.isna().sum()
# # Print the resulting DataFrame
# print(customer_orders)
# customer_orders.shape
# # Reset the index to convert CustomerId into a column
# customer_orders = customer_orders.reset_index()

# # Merge the CartTotal column into customer_orders based on the CustomerId column

# # Access the columns using indexing
# customer_id = customer_orders['CustomerId']
# cart_ids = customer_orders['CartID']
# cart_totals = customer_orders['CartTotal']
# df.CartID.nunique()
# df.TotalCost.nunique()

# df.columns

# df.isna().sum()
# import pandas as pd
# df.CustomerId.unique().shape
# # select 10 customers to focus on
# customers = df.CustomerId.unique()[:3]
# df.isna().sum()
# [['CustomerID','CartID']].isna().sum()
# df.shape
# customers
# # create a new dataframe to store the results
# results_df = pd.DataFrame(columns=['CustomerId', 'OrderDate', 'OrderFrequency', 'OrderValue', 'CategoryPopularity'])
# ten_cust = df.loc[df['CustomerId'].isin(customers)]

# cust_df = pd.DataFrame(columns=['CustomerId','OrderFrequency', 'OrderValue'])

# # loop through each customer and calculate their order frequency, order value, and category popularity
# for customer in customers:
#     # select orders for the current customer
#     customer_orders = df[df.CustomerId == customer]
#     customer_orders.columns
#     # calculate the order frequency
#     order_frequency = len(customer_orders) / len(df.CartID)
#     # print(order_frequency)
    
#     df.columns
    
    
#     # calculate the order value
#     order_value = customer_orders.TotalCost.sum()
#     # print(order_value)
#     cust_df = cust_df.append({
#         'CustomerId': customer,
#         'OrderFrequency': order_frequency,
#         'OrderValue': order_value,
#     }, ignore_index=True)
# customer_orders.shape
# # pickle the dataframe
# df.to_pickle('CustomerFreq.pkl')
# cust_df
#     # calculate the category popularity
#     category_counts = customer_orders.groupby('MenuItemID').size().reset_index(name='Count')
#     category_counts['Popularity'] = category_counts['Count'] / category_counts['Count'].sum()
#     category_popularity = category_counts.to_dict(orient='records')
    
#     # add the results to the dataframe
#     results_df = results_df.append({
#         'CustomerId': customer,
#         'OrderDate': customer_orders.DateCreated.min(),
#         'OrderFrequency': order_frequency,
#         'OrderValue': order_value,
#         'ItemPopularity': category_popularity
#     }, ignore_index=True)

# # print the results
# print(results_df)

# results_df.shape

# import pandas as pd

# # Load the data into a dataframe
# df = pd.read_csv('data.csv')

# # Create a dataframe ranking menu item popularity among all items ordered
# item_popularity_all = df.groupby('ItemName').size().sort_values(ascending=False).reset_index(name='PopularityAll')

# # Create a dataframe ranking menu item popularity by day of the week
# item_popularity_by_day = pd.melt(df[['ItemName', 'DateCreated', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']], 
#                                  id_vars=['ItemName', 'DateCreated'], 
#                                  value_vars=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], 
#                                  var_name='DayOfWeek', 
#                                  value_name='Count')
# item_popularity_by_day = item_popularity_by_day.dropna(subset=['Count'])
# item_popularity_by_day = item_popularity_by_day.groupby(['ItemName', 'DayOfWeek']).size().reset_index(name='Count')
# item_popularity_by_day = item_popularity_by_day.sort_values(['ItemName', 'DayOfWeek', 'Count'], ascending=[True, True, False])

# # Merge the two dataframes on 'ItemName'
# item_popularity = pd.merge(item_popularity_all, item_popularity_by_day, on='ItemName', how='outer')


# item_popularity.tail(10)

# item_counts = [x for x in item_counts]
# item_counts.hist()
# plt.show()
# # Create a list of labels for the bins using the item_dict
# labels = [item_dict.get(mid) for mid in item_counts.index if mid in item_dict]
# labels
# [type(l) for l in labels]
# # Create a bar chart using matplotlib
# fig, ax = plt.subplots(figsize=(10, 5))
# ax.barh(labels, item_counts, height=0.5, edgecolor='black')
# ax.set_xlabel('Count')
# ax.set_ylabel('Item Name')
# ax.set_title('Histogram of Items Ordered')
# len(labels),len(item_counts)
# plt.show()
# x = ['a','b']
# y = [1,2]
# fig, ax = plt.subplots(figsize=(10, 5))
# ax.hist(labels, item_counts, height=0.5, edgecolor='black')
# ax.set_xlabel('Count')
# ax.set_ylabel('Item Name')
# ax.set_title('Histogram of Items Ordered')
# plt.show()
# plt.hist()