import pandas as pd 
import numpy as np
import random
import pymongo

#---------------------------------------
#           Import the data 
#---------------------------------------

client = pymongo.MongoClient("mongodb://localhost:27017")
db = client[ "PFE" ]

#products
products = db[ "products" ]
products_df = pd.DataFrame(list(products.find()))
products_df = products_df.drop('_id', 1)

#like
likes = db[ "metauser" ]
tab = pd.DataFrame(list(likes.find()))
tab = tab.drop('_id', 1)
tab = tab.drop('rating', 1)
tab = tab[tab.like != "-"]

tab.columns
tab['userId'] = tab['userId'].astype(int)
#useri the id of the user, frequsers the freq of each user
useri,frequsers=np.unique(tab.userId,return_counts=True)
#itemi the item id, freqitem the freq of each item
itemi,freqitems=np.unique(tab.produitId,return_counts=True)
n_users=len(useri)
n_items=len(itemi)
print("le nombre des utilisateurs est :"+ str(n_users) + " Et le nombre des items est: "+ str(n_items))

#---------------------------------------
#      preprocessing the data 
#---------------------------------------
"""
One of the problems we encountered was the fact that the product ids were not ordered.
That is to say we can find user 1,2,3,5 and 8 without finding users 4, 6 and 7.
This gave us a problem in creating the user-product matrix because we risk having several empty rows and columns.
To do this, we created an index_user array and an index_product array which contain the old id's and the new id's eg (1,2,5,6) => (1,2,3,4)
then we added two columns on the main table which contains these new IDs.
we exported the new ids to a csv file, and each time we use this new file.
"""

indice_user = pd.DataFrame()
indice_user["indice"]=range(1,len(useri)+1)
indice_user["useri"]=useri

indice_item = pd.DataFrame()
indice_item["indice"]=range(1,len(itemi)+1)
indice_item["itemi"]=itemi
 
#create user_ID_new and Item_ID_new
x=[]
y=[]

for i in range(0,len(tab)):
    x.append((indice_user.indice[indice_user.useri==tab.userId.iloc[i]].axes[0]+1)[0])
    y.append((indice_item.indice[indice_item.itemi==tab.produitId.iloc[i]].axes[0]+1)[0])

tab["userIdnew"]=x
tab["produitIdnew"]=y

tab[:20]

data_matrix = np.zeros((n_users, n_items))
for line in tab[["userIdnew","produitIdnew","like"]].itertuples():
    data_matrix[line[1]-1, line[2]-1] = line[3]*random.randint(5,10)
    
#functions to pass from old to new Id
def change_item_to_newId(id):
    return tab.produitIdnew[tab.produitId==id].iloc[0]
def change_item_to_Id(id):
    return tab.produitId[tab.produitIdnew==id].iloc[0]

def change_user_to_newIduser(id):
    return tab.userIdnew[tab.userId==id].iloc[0]
def change_user_to_Iduser(id):
    return tab.userId[tab.userIdnew==id].iloc[0]

#---------------------------------------
#      ALS : Alternating Least Squares
#---------------------------------------

    
#Index matrix for training data, if a product liked by a user (1: yes, 0: no)
I = data_matrix.copy()
I[I > 0] = 1
I[I == 0] = 0

lmbda = 0.1 # Regularization term
k = 20 # dimension of the hidden character space
n_epochs = 2 
m, n = data_matrix.shape # Number of users and items
# Latent user feature matrix : # Matrix of hidden characters for users
U = np.random.rand(k,m) 
# Latent item feature matrix : # Matrix of hidden characters for items
P = np.random.rand(k,n) 
P[0,:] = data_matrix[data_matrix != 0].mean(axis=0) # Avg. rating for each product
#the U and P matrices are initialized with random values at the start, but their content changes with each iteration based on the train
E = np.eye(k) # (k x k)-dimensional idendity matrix


#RMSE = RacineCarrée{(1/N) * sum (r_i -estimé{r_i})^2}
def rmse2(I,R,P,U):
    return np.sqrt(np.sum((I * (R - prediction(U,P)))**2)/len(R[R > 0]))

# The prediction function allows to predict the unknown ratings by multiplying the matrices the transpose of U and P
def prediction(U,P):
    return np.dot(U.T,P)

train_errors = []
test_errors = []

# Repeat until convergence
for epoch in range(n_epochs):
    # Fix P and estimate U
    for i, Ii in enumerate(I):
        nui = np.count_nonzero(Ii) # Number of items user i has rated
    
        # Least squares solution
        Ai = np.dot(P, np.dot(np.diag(Ii), P.T)) + lmbda * nui * E
        Vi = np.dot(P, np.dot(np.diag(Ii), data_matrix[i].T))
        U[:,i] = np.linalg.solve(Ai,Vi)
        
    # Fix U and estimate P
    for j, Ij in enumerate(I.T):
        nmj = np.count_nonzero(Ij) # Number of users that rated item j
        
        # Least squares solution
        Aj = np.dot(U, np.dot(np.diag(Ij), U.T)) + lmbda * nmj * E
        Vj = np.dot(U, np.dot(np.diag(Ij), data_matrix[:,j]))
        P[:,j] = np.linalg.solve(Aj,Vj)
    
    train_rmse = rmse2(I,data_matrix,P,U)
    train_errors.append(train_rmse)
    
    
    print ("[Epoch %d/%d] train error: %f" \
    %(epoch+1, n_epochs, train_rmse))
    
print ("Algorithm converged")

model_matrix=prediction(U,P) #matrice (user,item)
model_matrix[model_matrix < 4.5] = 0
model_matrix[model_matrix >= 4.5] = 1
model_df=pd.DataFrame(model_matrix)

model_df

def als_recom_it_for_user(id):
    id=id-1
    similar_indices = model_matrix[id].argsort()[:-250:-1]
    #model_matrix [id] the rates for a user
    similar_items = [(model_matrix[id][i], change_item_to_Id(i+1)) for i in similar_indices]
    #ps: i+1 to eliminate the effect of starting with a zero our list of items!
    return similar_items[:]

recommended = []
for i in range(0,n_users):
    
    recommand = als_recom_it_for_user(i)
    df = pd.DataFrame(recommand, columns =['like', 'produitId']) 
    rv = df.to_json(orient='records')
    recommended.append(rv)
    
resultat = pd.DataFrame(recommended)

resultat['userId'] = resultat.index+1

for i in range(0,n_users-1):
    resultat["userId"][i] = change_user_to_Iduser(i+1)
    
resultat.to_csv(r'C:/Users/abd/OneDrive/Bureau/PFE_FILES_FIN/ALSLikes.csv')