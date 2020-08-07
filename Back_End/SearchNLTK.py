import pandas as pd
import numpy as np
from rake_nltk import Rake
import pymongo
client = pymongo.MongoClient("mongodb://localhost:27017")
db = client[ "PFE" ]

#products
products = db[ "products" ]
produits = pd.DataFrame(list(products.find()))
produits = produits.drop('_id', 1)

df = produits[['Product_name']]
df['Product_name'] = df['Product_name'].astype(str)
    
df['Product_name'] = df['Product_name'].map(lambda x: x.split(' '))

for index, row in df.iterrows():
    df['Product_name'][index] = [x.lower() for x in row['Product_name']]
    
df['Bag_of_words'] = ''

columns = ['Product_name']
for index, row in df.iterrows():
    words = ''
    for col in columns:
        words += ' '.join(row[col]) + ''
    df['Bag_of_words'][index] = words
    
df5 = df[['Bag_of_words']]

df5['Bag_of_words'] = df5['Bag_of_words'].astype(str)

a = np.asarray(df5['Bag_of_words'])
print(a)
np.savetxt('C:/Users/abd/OneDrive/Bureau/PFE_FILES_FIN/test.txt', a,fmt="%a") 