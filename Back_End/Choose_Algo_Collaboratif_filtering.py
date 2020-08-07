import pandas as pd 
import numpy as np
import random
import pymongo


#---------------------------------------
#           Import the data 
#---------------------------------------
client = pymongo.MongoClient("mongodb://localhost:27017")
db = client[ "PFE" ]

#rating
ratings = db[ "metauser" ]
tab = pd.DataFrame(list(ratings.find()))
tab = tab.drop('_id', 1)
tab = tab.drop('like', 1)
tab = tab[tab.rating != "-"]

tab.columns

tab['userId'] = tab['userId'].astype(int)
useri,frequsers=np.unique(tab.userId,return_counts=True)
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

#cross validation
from sklearn import model_selection as cv
train_data, test_data = cv.train_test_split(tab[["userIdnew","produitIdnew","rating"]], test_size=0.25,random_state=123)

 
# The rule says that if we have a great sparsity (ie it is not to be able to calculate the similarity between 2 users 
# eg if each one liked different items than the other), the models of Model Based will be the most effective.
# Then let's us calculate the sparsity:

sparsity=round(1.0-len(tab)/float(n_users*n_items),3)
print('The sparsity level of our data base is ' +  str(sparsity*100) + '%')

#The percentage of sparsity is very large therefore, we can now confirm that the Model Based models will be the most efficient models.

#====================================================================================================
#                                   1. Memory based Collaboratif Filtering
#====================================================================================================
# ### 1.1 Setting up the model:
# We will start by creating the Memory based models.
# -User-Based models: "Users who are similar to you also liked ..."
# -The Item-Based models: "Users who liked this also liked ..."
# To explain more:
# -the User-Based model: will take a user, find the users most similar to him based on the rating, then recommend the items liked by these users (it takes a user and returns items)
# -The Item-Based model: takes an item, searches for users who liked this item, finds items liked by these users
# (takes an item and returns a list of items)
# To do this, we use 2 metrics the similar cosine and cityblock.
# To do this, we starte by creating the user-item train and test matrix. These are the two matrices that will cross the user and item notes.
# Then, we create our 4 Memory Based models
# at the end, we create a function to make the predictions according to the model

train_data_matrix = np.zeros((n_users, n_items))# null matrix of length of all users and all items
for line in train_data.itertuples():##browse each row , col by col
    train_data_matrix[line[1]-1, line[2]-1] = line[3]

test_data_matrix = np.zeros((n_users, n_items))
for line in test_data.itertuples():
    test_data_matrix[line[1]-1, line[2]-1] = line[3]

#calculate cos similarity: (model construction)
from sklearn.metrics.pairwise import pairwise_distances
user_similarity = pairwise_distances(train_data_matrix, metric='cosine')
item_similarity = pairwise_distances(train_data_matrix.T, metric='cosine')
user_similarity1 = pairwise_distances(train_data_matrix, metric='cityblock')
item_similarity1 = pairwise_distances(train_data_matrix.T, metric='cityblock')

def predict(ratings, similarity, type='user'):
    if type == 'user':
        mean_user_rating = ratings.mean(axis=1)#mean for each user (type = float)
       
        ratings_diff = (ratings - mean_user_rating[:, np.newaxis]) 
        pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
    elif type == 'item':
        pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
        
    x = np.zeros((n_users, n_items))
    for i in range(0,n_items):
        a=max(pred[:,i])
        b=min(pred[:,i])
        c=0 # min rating
        d=5 # max rating
        for j in range(0,n_users):
            x[j,i]=(pred[:,i][j]-(a-c))*d/(b-a+c)
    
    return x

#the prediction with the different models:
item_prediction = predict(test_data_matrix, item_similarity, type='item')
user_prediction = predict(test_data_matrix, user_similarity, type='user')
item_prediction1 = predict(test_data_matrix, item_similarity1, type='item')
user_prediction1 = predict(test_data_matrix, user_similarity1, type='user')

#1.2. The comparison of RMSE:
#the creation of the function which calculates the RMSE:
from sklearn.metrics import mean_squared_error
from math import sqrt
def rmse(prediction, ground_truth): #Root Mean Squared Error
    prediction = prediction[ground_truth.nonzero()].flatten() 
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    return sqrt(mean_squared_error(prediction, ground_truth))

print ('User-based CF: The RMSE for the cosine similarity metric is : ' + str(rmse(user_prediction, test_data_matrix)))
print ('Item-based CF: The RMSE for the cosine similarity metric is :  ' + str(rmse(item_prediction, test_data_matrix)))
print ('User-based CF: The RMSE for the cityblock similarity metric is :   ' + str(rmse(user_prediction1, test_data_matrix)))
print ('Item-based CF: The RMSE for the cityblock similarity metric is : ' + str(rmse(item_prediction1, test_data_matrix)))

# The best model is the one with the smallest RMSE.
# In our case it was User based for the cosine metric.
"""
# Conclusion: 
------------    
# -Memory based models are easy to implement and generate good results.
# -This type of model is not scalable (is not really practical in a problem of a large database since it
# (when starting with a new user / item of which we do not have enough information) calculate the correlation between all the users / items each time) and do not solve the cold start problem
# To answer the problem of scalability we create the Model Based models (next part).
# To answer the cold start problem, we use the Content based recommendation (we will not use it since we don't have this data)
"""

#====================================================================================================
#                                   # 2. Model-based Collaborative Filtering
#====================================================================================================
# In this part of the project, we apply the second sub-type of collaborative filtering: "Model-based".
# It consists in applying the matrix factorization (MF): it is an unsupervised learning method of decomposition and 
# dimensionality reduction for hidden variables.
# The purpose of the matrix factorization is to learn the hidden preferences of users and the hidden attributes of items
# from known ratings in our dataset, to finally predict unknown ratings by multiplying the hidden varibal matrices of users and items.
# There are several dimensionality reduction techniques in the implementation of recommendation systems.

# In our project , we used:
# - SVD: singular value decomposition
# - SGD: Stochastic Gradient Descent
# - ALS: Alternating Least Squares

#-----------------------------------------------------
#      2.1 Singular value decomposition (SVD)
#-----------------------------------------------------
# 2.1.1 The implementation of SVDs:
# This technique, like all the others, consists in reducing the dimensionality of the User-Item matrix calculated previously.
# Let R be the User-Item matrix of size m x n (m: number of users, n: number of items) and k: the dimension of the space of hidden characters.
# The general equation of SVD is given by: R = USV ^ T with:
# - The matrix U of hidden characters for users: of size m * k
# - The matrix V of hidden characters for items: of size n * k
# - The diagonal matrix of size k x k with non-negative real values on the diagonal
# We can make the prediction by applying the multiplication of the 3 matrices

from scipy.sparse.linalg import svds

#Get the components of SVD from the User-Item matrix of the train. We choose a value of k = 20.
u, s, vt = svds(train_data_matrix, k = 20)
s_diag_matrix=np.diag(s)

# Multiplication of the 3 matrices with np.dot to obtain the estimated User_Item matrix.
X_pred = np.dot(np.dot(u, s_diag_matrix), vt) 

# the normalization of X_pred since it returns data which is not well distributed in [0,5]
import math
x = np.zeros((n_users, n_items))
for i in range(0,n_items):
    a=max(X_pred[:,i])
    b=min(X_pred[:,i])
    c=0
    d=5
    for j in range(0,n_users):
        x[j,i]=(X_pred[:,i][j]-(a-c))*d/(b-a+c)
        if math.isnan(x[j,i]): x[j,i]=0

# Performance calculation with RMSE between the estimated matrix and the test matrix
print ('RMSE: ' + str(rmse(x, test_data_matrix)))

"""
conclusion:
-----------
# We found 1.4610559480936944 as RMSE, it's bigger than the RMSE of memory based models, but it takes a lot less time.
#What we are going in the next part is to improve our model by the stochastic gradient and the ALS.
"""

#---------------------------------------------------------
#      2.2 Algorithme SGD (Stochastic Gradient Descent)
#---------------------------------------------------------
#  When we use collborative filtering for SGD, we want to estimate 2 matrices U and P:
# - The matrix U of hidden characters for users: of size m * k (m: number of users, k: dimension of the space of hidden characters)
# - The matrix P of hidden characters for items: of size n * k (m: number of items, k: dimension of the space of hidden characters)
# After estimating U and P, we can then predict the unknown ratings by multiplying the matrices the transpose of U and P.

# The I and I2 matrices will be used as selector matrices to take the appropriate elements after the creation of the Train and the Test

# matrix of indices for the train
I = train_data_matrix.copy()
I[I > 0] = 1
I[I == 0] = 0

# matrix of indices for the test
I2 = test_data_matrix.copy()
I2[I2 > 0] = 1
I2[I2 == 0] = 0

# The prediction function allows to predict the unknown ratings by multiplying the matrices the transpose of U and P
def prediction(U,P):
    return np.dot(U.T,P)
 
# To update U and P, we can use the SGD where we iterate each observation in the train to update U and P from time to time:
# P_{i+1} = P_i + (gamma) (e{ui}*U_u - (lambda)* P_i)
# U_{i+1} = U_i + (gamma) (e{ui}*P_u - (lambda)* U_i)
    
# we note: 
# - (gamma) the speed of learning
# - (lambda) the Term of Regularization 
# - e: the error which is the difference between the actual rating and the predicted rating.

# ****** Initialisation ******* 
lmbda = 0.1 
k = 20 
m, n = train_data_matrix.shape  
steps = 150  
gamma=0.001  
U = np.random.rand(k,m)
P = np.random.rand(k,n) 


def rmse2(I,R,P,U):
    return np.sqrt(np.sum((I * (R - prediction(U,P)))**2)/len(R[R > 0]))


# We only consider the values! = 0
users,items = train_data_matrix.nonzero()  

# SGD implementation: (ps) this algo takes a long time depending on the number of steps chosen.
train_errors = [] # store the train errors obtained by RMSE at each iteration (step)
test_errors = [] # store the test errors obtained by RMSE at each iteration (step)
     
for step in range(steps):
    for u, i in zip(users,items): 
        e = train_data_matrix[u, i] - prediction(U[:,u],P[:,i])  # calculate the error e for the gradient
        U[:,u] += gamma * ( e * P[:,i] - lmbda * U[:,u]) # update of the matrix U
        P[:,i] += gamma * ( e * U[:,u] - lmbda * P[:,i]) # update of the  matrix P
        
    train_rmse = rmse2(I,train_data_matrix,P,U) # Calculation of the RMSE from the train
    test_rmse = rmse2(I2,test_data_matrix,P,U) # Calculation of the RMSE from the test
    train_errors.append(train_rmse) # at each iteration add the error to the list
    test_errors.append(test_rmse) 

print('RMSE : ' + str(np.mean(test_errors)))

# Now, after getting all the error values at each step, we can draw the learning curve.
# ==> We check the performance by tracing the train and test errors

import matplotlib.pyplot as plt

plt.plot(range(steps), train_errors, marker='o', label='Training Data'); 
plt.plot(range(steps), test_errors, marker='v', label='Test Data');
plt.title('Courbe d apprentissage SGD')
plt.xlabel('Nombre d etapes');
plt.ylabel('RMSE');
plt.legend()
plt.grid()
plt.show()

"""
conclusion:
-----------
# The model seems working well with relatively low RMSE value after convergence.
# The performance of the model may depend on the parameters (gamma), (lambda) and k that we have varied several times in order to obtain
# the best RMSE.
# After this step, we can compare the actual rating with the estimated rating; To do this, we use the User-item matrix that we have
# already calculated and used the prediction function (U, P) implemented previously.
"""

#------------------------------------------------
#     2.3 ALS : Alternating Least Squares
#------------------------------------------------

# Index matrix for training data
I = train_data_matrix.copy()
I[I > 0] = 1
I[I == 0] = 0

# Index matrix for test data
I2 = test_data_matrix.copy()
I2[I2 > 0] = 1
I2[I2 == 0] = 0

lmbda = 0.1 
k = 20 
n_epochs = 2 
m, n = test_data_matrix.shape # Number of users and items
U = np.random.rand(k,m) # Latent user feature matrix : # Matrice des caractéres cachés pour les utilisateurs
P = np.random.rand(k,n) # Latent item feature matrix : # Matrix of hidden characters for items
P[0,:] = test_data_matrix[test_data_matrix != 0].mean(axis=0) # Avg. rating for each product
#the U and P matrices are initialized with random values at the start, but their content changes with each iteration based on the train

E = np.eye(k) 
train_errors = []
test_errors = []

#RMSE = RacineCarrée{(1/N) * sum (r_i -estimé{r_i})^2}
def rmse2(I,R,P,U):
    return np.sqrt(np.sum((I * (R - prediction(U,P)))**2)/len(R[R > 0]))

# Repeat until convergence
for epoch in range(n_epochs):
    # Fix P and estimate U
    for i, Ii in enumerate(I):
        nui = np.count_nonzero(Ii) # Number of items user i has rated
    
        # Least squares solution
        Ai = np.dot(P, np.dot(np.diag(Ii), P.T)) + lmbda * nui * E
        Vi = np.dot(P, np.dot(np.diag(Ii), train_data_matrix[i].T))
        U[:,i] = np.linalg.solve(Ai,Vi)
        
    # Fix U and estimate P
    for j, Ij in enumerate(I.T):
        nmj = np.count_nonzero(Ij) # Number of users that rated item j
        
        # Least squares solution
        Aj = np.dot(U, np.dot(np.diag(Ij), U.T)) + lmbda * nmj * E
        Vj = np.dot(U, np.dot(np.diag(Ij), train_data_matrix[:,j]))
        P[:,j] = np.linalg.solve(Aj,Vj)
    
    train_rmse = rmse2(I,train_data_matrix,P,U)
    test_rmse = rmse2(I2,test_data_matrix,P,U)
    train_errors.append(train_rmse)
    test_errors.append(test_rmse)
    
    print ("[Epoch %d/%d] train error: %f, test error: %f"     %(epoch+1, n_epochs, train_rmse, test_rmse))
    
print ("Algorithm converged")

"""
Conclusion
----------
This algorithm is the best of all other algorithms. In the second iteration we found a train error which is equal to 0.773646 and a test error which is equal to 1.273079
As this is the fastest and most efficient algorithm, we decided to generalize it over the whole dataset.
"""