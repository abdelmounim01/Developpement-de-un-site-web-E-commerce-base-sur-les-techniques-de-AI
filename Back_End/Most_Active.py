import numpy as np
import pandas as pd
import pymongo
client = pymongo.MongoClient("mongodb://localhost:27017")
db = client[ "PFE" ]

#products
users = db[ "users" ]
print(users)
dfusers = pd.DataFrame(list(users.find()))
dfusers = dfusers.drop('_id', 1)

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

#compute the count votes , likes and mean value as group by the users
count = dfrating.groupby("userId", as_index=False).count()
mean = dfrating.groupby("userId", as_index=False).mean()
countlikes = dflike.groupby("userId", as_index=False).count()

#merge two dataset create df1
df1 = pd.merge(dfrating, count, how='right', on=['userId'])

#merge two dataset create df7
df7 = pd.merge(df1, mean, how='right', on=['userId'])

#merge two dataset create df6
df6 = pd.merge(df7, countlikes, how='right', on=['userId'])

#rename column
df6["Count"] = df6["rating_y"]
df6["Rating"] = df6["rating_x"]
df6["Mean"] = df6["rating"]

#Create New dataframe with selected variables
df6 = df6[['userId',"Count","like","Mean"]]
#select distinct
df6 = df6.drop_duplicates()

#sorted users
df2 = df6.sort_values(by=['Count','like','Mean'], ascending=False)

df9 = pd.merge(df2, dfusers, how='right', on=['userId'])
df9.loc[df9['Count'].isnull(), 'Count'] = 0
df9.loc[df9['Mean'].isnull(), 'Mean'] = 0
df9.loc[df9['like'].isnull(), 'like'] = 0

#20 Most Active Users
df8 = df9.head(20)

#20 least Active Users
df10 = df9.tail(20)

df8.to_csv(r'C:/Users/abd/OneDrive/Bureau/PFE_FILES_FIN/MostActive.csv')
df10.to_csv(r'C:/Users/abd/OneDrive/Bureau/PFE_FILES_FIN/LeastActive.csv')