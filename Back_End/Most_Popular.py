import numpy as np
import pandas as pd
import pymongo
client = pymongo.MongoClient("mongodb://localhost:27017")
db = client[ "PFE" ]

#products
products = db[ "products" ]
print(products)
dfproducts = pd.DataFrame(list(products.find()))
dfproducts = dfproducts.drop('_id', 1)

#like
likes = db[ "metauser" ]
dflike = pd.DataFrame(list(likes.find()))
dflike = dflike.drop('_id', 1)
dflike = dflike.drop('rating', 1)
dflike = dflike[dflike.like != "-"]
dflike['like'] = dflike['like'].astype(np.int)

#rating
ratings = db[ "metauser" ]
dfrating = pd.DataFrame(list(ratings.find()))
dfrating = dfrating.drop('_id', 1)
dfrating = dfrating.drop('like', 1)
dfrating = dfrating[dfrating.rating != "-"]
dfrating['rating'] = dfrating['rating'].astype(np.int)

#compute the count and mean value as group by the products
count = dfrating.groupby("produitId", as_index=False).count()
mean = dfrating.groupby("produitId", as_index=False).mean()
countlikes = dflike.groupby("produitId", as_index=False).count()

#merge two dataset create df1
df1 = pd.merge(dfrating, count, how='right', on=['produitId'])
#merge two dataset create df7
df7 = pd.merge(df1, mean, how='right', on=['produitId'])
#merge two dataset create df6
df6 = pd.merge(df7, countlikes, how='right', on=['produitId'])

#rename column
df6["Count"] = df6["rating_y"]
df6["Rating"] = df6["rating_x"]
df6["Mean"] = df6["rating"]

#Create New datafram with selected variables
df6 = df6[['produitId',"Count","Mean","like"]]

#select distinct
df6 = df6.drop_duplicates()

#sorted products
df2 = df6.sort_values(by=['Mean','like','Count',], ascending=False)

df9 = pd.merge(df2, dfproducts, how='right', on=['produitId'])
df9.loc[df9['Count'].isnull(), 'Count'] = 0
df9.loc[df9['Mean'].isnull(), 'Mean'] = 0
df9.loc[df9['like'].isnull(), 'like'] = 0

#20 Most Popular Products
df8 = df9.head(20000)

#20 least Popular Products
df10 = df9.tail(20)

df8.to_csv(r'C:/Users/abd/OneDrive/Bureau/PFE_FILES_FIN/MostPopular.csv')
df9.to_csv(r'C:/Users/abd/OneDrive/Bureau/PFE_FILES_FIN/Popularityproducts.csv')
df10.to_csv(r'C:/Users/abd/OneDrive/Bureau/PFE_FILES_FIN/LeastPopular.csv')