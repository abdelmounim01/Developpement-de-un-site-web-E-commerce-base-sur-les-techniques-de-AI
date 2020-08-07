import pandas as pd

ALSRatings = pd.read_csv('C:/Users/abd/OneDrive/Bureau/PFE_FILES_FIN/ALSRatings.csv')
ALSRatings.rename(columns={ALSRatings.columns[1]: "rating"}, inplace = True)
ALSRatings = ALSRatings.drop('Unnamed: 0', 1)

ALSLikes = pd.read_csv('C:/Users/abd/OneDrive/Bureau/PFE_FILES_FIN/ALSLikes.csv')
ALSLikes.rename(columns={ ALSLikes.columns[1]: "like" }, inplace = True)
ALSLikes = ALSLikes.drop('Unnamed: 0', 1)

resultat = pd.merge(ALSRatings,ALSLikes , how='right', on=['userId'])
resultat["like"].fillna("", inplace=True)
resultat["rating"].fillna("", inplace=True)
resultat['Maylike'] = resultat['rating'].add(resultat['like'])

for index,row in resultat.iterrows() :
    lgenre=(resultat.ix[index,'Maylike']).split('][')
    resultat.ix[index,'Maylike']=',' .join(lgenre)

resultat = resultat[['userId','Maylike']]
resultat.to_csv(r'C:/Users/abd/OneDrive/Bureau/PFE_FILES_FIN/ALSRatingsLikes.csv')
