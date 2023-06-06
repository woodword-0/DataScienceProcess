import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
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

# Get the top 5 most frequent items
top_items = df['ItemName'].value_counts().nlargest(5).index.unique()
top_items
# Filter the data to include only the top 5 items
filtered_data = df[df['ItemName'].isin(top_items)]

# Join the item names into a single string
text = ' '.join(filtered_data['ItemName'])

# Create the word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

# Display the word cloud
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()