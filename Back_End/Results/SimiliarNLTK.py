from rake_nltk import Rake
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer 
import pymongo
client = pymongo.MongoClient("mongodb://localhost:27017")
db = client[ "PFE" ]

#products
products = db[ "products" ]
products_df = pd.DataFrame(list(products.find()))
products_df = products_df.drop('_id', 1)

#like
likes = db[ "metauser" ]
like = pd.DataFrame(list(likes.find()))
like = like.drop('_id', 1)
like = like.drop('rating', 1)
like = like[like.like != "-"]

#rating
ratings = db[ "metauser" ]
rating = pd.DataFrame(list(ratings.find()))
rating = rating.drop('_id', 1)
rating = rating.drop('like', 1)
rating = rating[rating.rating != "-"]

rating['rating'] = rating['rating'].astype(float)
rating = rating.groupby('produitId', as_index=False)['rating'].mean()
like['like'] = like['like'].astype(float)
like = like.groupby('produitId', as_index=False)['like'].count()

products_df = pd.merge(products_df, rating , on='produitId' , how='left')
products_df = pd.merge(products_df, like , on='produitId' , how='left')
products_df.loc[products_df['rating'].isnull(), 'rating'] = "0"
products_df.loc[products_df['like'].isnull(), 'like'] = "0"
products_df['rating'] = products_df['rating'].astype(float)
products_df['like'] = products_df['like'].astype(float)
products_df['rating'] = products_df['rating'].round(1)
products_df['description'] = products_df['description'].astype(str)

#------------------------------------------------------------------------------------------
#Traitement Pour Nom et Description

df = products_df[['Product_name',"Categories","description","rating","like"]]
df['Product_name'] = df['Product_name'].astype(str)
df['Categories'] = df['Categories'].astype(str)
df['description'] = df['description'].astype(str)
df['rating'] = df['rating'].astype(str)
df['like'] = df['like'].astype(str)

df['Key_words'] = ''

r = Rake()
for index, row in df.iterrows():
    r.extract_keywords_from_text(row['description'])
    key_words_dict_scores = r.get_word_degrees()
    df['Key_words'][index] = list(key_words_dict_scores.keys())
    
df['Product_name'] = df['Product_name'].map(lambda x: x.split(' '))

for index, row in df.iterrows():
    df['Product_name'][index] = [x.lower() for x in row['Product_name']]
    
    
df['Bag_of_words'] = ''
columns = ['Product_name','Key_words']

for index, row in df.iterrows():
    words = ''
    for col in columns:
        words += ' '.join(row[col]) + ''
    df['Bag_of_words'][index] = words
    
df1 = df[['Product_name','Bag_of_words']]

df1['Bag_of_words'] = df1['Bag_of_words'].astype(str)

count = CountVectorizer()
count_matrix = count.fit_transform(df1['Bag_of_words'])
cosine_sim1 = cosine_similarity(count_matrix, count_matrix)

#------------------------------------------------------------------------------------------
#Traitement Pour Genre

df = products_df[['Product_name',"Categories","subCategory","description","rating","like"]]
df['Product_name'] = df['Product_name'].astype(str)
df['Categories'] = df['Categories'].astype(str)
df['subCategory'] = df['subCategory'].astype(str)
df['description'] = df['description'].astype(str)
df['rating'] = df['rating'].astype(str)
df['like'] = df['like'].astype(str)

df['Categories'] = df['Categories'].map(lambda x: x.split(' '))

for index, row in df.iterrows():
    df['Categories'][index] = [x.lower() for x in row['Categories']]
       
df['Bag_of_words'] = ''
columns = ['Categories']

for index, row in df.iterrows():
    words = ''
    for col in columns:
        words += ' '.join(row[col]) + ''
    df['Bag_of_words'][index] = words
    
df2 = df[['Product_name','Bag_of_words']]

df2['Bag_of_words'] = df2['Bag_of_words'].astype(str)

count = CountVectorizer()
count_matrix = count.fit_transform(df2['Bag_of_words'])
cosine_sim2 = cosine_similarity(count_matrix, count_matrix)

#------------------------------------------------------------------------------------------
#Traitement Pour Score

df = products_df[['Product_name',"Categories","description","rating","like"]]
df['Product_name'] = df['Product_name'].astype(str)
df['Categories'] = df['Categories'].astype(str)
df['description'] = df['description'].astype(str)
df['rating'] = df['rating'].astype(np.float64)
df['like'] = df['like'].astype(str)

for i in range(len(df)):
    b = df['rating'][i]
    df['rating'][i] = round(b)

#transforme float to lettre for use in nltk
for i in range(len(df)):
    if df['rating'][i] == 0:
        df['rating'][i] = 'Zero'
    if df['rating'][i] == 1:
           df['rating'][i] = 'Zero One'
    if df['rating'][i] == 2:
           df['rating'][i] = 'Zero One Two'
    if df['rating'][i] == 3:
           df['rating'][i] = 'Zero One Two Three'
    if df['rating'][i] == 4:
           df['rating'][i] = 'Zero One Two Three Four'
    if df['rating'][i] == 5:
           df['rating'][i] = 'Zero One Two Three Four Five'

df['rating'] = df['rating'].map(lambda x: x.split(' '))

df['Bag_of_words'] = ''
columns = ['rating']

for index, row in df.iterrows():
    words = ''
    for col in columns:
        words += ' '.join(row[col]) + ''
    df['Bag_of_words'][index] = words
    
df3 = df[['Product_name','Bag_of_words']]

df3['Bag_of_words'] = df3['Bag_of_words'].astype(str)

count = CountVectorizer()
count_matrix = count.fit_transform(df3['Bag_of_words'])
cosine_sim3 = cosine_similarity(count_matrix, count_matrix)

#------------------------------------------------------------------------------------------
#Traitement Pour Like

df = products_df[['Product_name',"Categories","description","rating","like"]]
df['Product_name'] = df['Product_name'].astype(str)
df['Categories'] = df['Categories'].astype(str)
df['description'] = df['description'].astype(str)
df['rating'] = df['rating'].astype(np.float64)
df['like'] = df['like'].astype(np.int)

#transforme 1
for i in range(len(df)):
    if df['like'][i] <= 10 :
        df['like'][i] = 0
    if df['like'][i] > 10 and df['like'][i] <= 30  :
           df['like'][i] = 1
    if df['like'][i] > 30 and df['like'][i] <= 50 :
           df['like'][i] = 2
    if df['like'][i] > 50 and df['like'][i] <= 100 :
           df['like'][i] = 3
    if df['like'][i] > 100 :
           df['like'][i] = 4
        
#transforme 2
for i in range(len(df)):
    if df['like'][i] == 0:
        df['like'][i] = 'So Bad'
    if df['like'][i] == 1:
           df['like'][i] = 'So Bad Bad'
    if df['like'][i] == 2:
           df['like'][i] = 'So Bad Bad Moyen'
    if df['like'][i] == 3:
           df['like'][i] = 'So Bad Bad Moyen Good'
    if df['like'][i] == 4:
           df['like'][i] = 'So Bad Bad Moyen Good Very Good'

df['like'] = df['like'].map(lambda x: x.split(' '))

df['Bag_of_words'] = ''
columns = ['like']

for index, row in df.iterrows():
    words = ''
    for col in columns:
        words += ' '.join(row[col]) + ''
    df['Bag_of_words'][index] = words
    
df4 = df[['Product_name','Bag_of_words']]

df4['Bag_of_words'] = df4['Bag_of_words'].astype(str)

count = CountVectorizer()
count_matrix = count.fit_transform(df4['Bag_of_words'])
cosine_sim4 = cosine_similarity(count_matrix, count_matrix)

#-------------------------------------------------------------------------
#Traitement Pour Subcategorie

df = products_df[['Product_name',"Categories","subCategory","description","rating","like"]]
df['Product_name'] = df['Product_name'].astype(str)
df['Categories'] = df['Categories'].astype(str)
df['subCategory'] = df['subCategory'].astype(str)
df['description'] = df['description'].astype(str)
df['rating'] = df['rating'].astype(str)
df['like'] = df['like'].astype(str)

df['subCategory'] = df['subCategory'].map(lambda x: x.split(' '))

for index, row in df.iterrows():
    df['subCategory'][index] = [x.lower() for x in row['subCategory']]
       
df['Bag_of_words'] = ''
columns = ['subCategory']

for index, row in df.iterrows():
    words = ''
    for col in columns:
        words += ' '.join(row[col]) + ''
    df['Bag_of_words'][index] = words
    
df5 = df[['Product_name','Bag_of_words']]

df5['Bag_of_words'] = df5['Bag_of_words'].astype(str)

count = CountVectorizer()
count_matrix = count.fit_transform(df5['Bag_of_words'])
cosine_sim5 = cosine_similarity(count_matrix, count_matrix)

#-------------------------------------------------------------------------
#calcule probabilité total et prendre 10 Similiaire Products pour Chaque Product

cosine_sim = np.zeros(shape=(len(products_df),len(products_df)))

for i in range(len(products_df)-1):
    for j in range(len(products_df)-1):
        cosine_sim[i][j] = ((cosine_sim1[i][j] * 4) + cosine_sim3[i][j]/2 + cosine_sim4[i][j]/2 + cosine_sim5[i][j] + cosine_sim2[i][j])/7

Dataframe = pd.DataFrame(data=cosine_sim[:,:])
Similiar = []

for i in range(len(products_df)-1):
    N = Dataframe[i]
    resultat = pd.DataFrame(N)
    resultat["produitId"] = products_df.produitId.loc[resultat.index == products_df.index]
    resultat = resultat.rename(columns={i: "Similarity"})
    resultat['produitId']=resultat['produitId'].astype(int)
    resultat=resultat[resultat.index != i]
    resultat = resultat.sort_values(by='Similarity',ascending=False)
    resultat = pd.merge(resultat, products_df, how='left', on=['produitId']).head(10)
    rv = resultat.to_json(orient='records')
    Similiar.append([rv])
    
print(Similiar[0])

from pyexcel_xlsx import save_data
save_data("C:/Users/errza/OneDrive/Bureau/PFE_FILES_FIN/silimiarproducts.xlsx",Similiar)