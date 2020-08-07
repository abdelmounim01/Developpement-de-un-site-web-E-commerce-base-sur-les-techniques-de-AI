import pandas as pd
import numpy as np
from flask import Flask, jsonify, request
from datetime import datetime
import os
import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
from tensorflow.keras.models import load_model
import json
import random
from keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from flask_pymongo import PyMongo
from flask_cors import CORS
from werkzeug.utils import secure_filename
from sklearn.utils import shuffle

app = Flask(__name__)
app.config['MONGO_DBNAME'] = 'PFE'
app.config['MONGO_URI'] = 'mongodb://localhost:27017/PFE'
mongo = PyMongo(app)
CORS(app)
UPLOAD_FOLDER = "C:/Users/abd/OneDrive/Bureau/Projet_FIN/src/assets/images/upload"
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

RatingsLikes = pd.read_csv('C:/Users/abd/OneDrive/Bureau/PFE_FILES_FIN/ALSRatingsLikes.csv')
RatingsLikes = RatingsLikes.drop('Unnamed: 0', 1)

Popular = pd.read_csv('C:/Users/abd/OneDrive/Bureau/PFE_FILES_FIN/Popularityproducts.csv')
Popular = Popular.drop('Unnamed: 0', 1)

Mostpopular = pd.read_csv('C:/Users/abd/OneDrive/Bureau/PFE_FILES_FIN/MostPopular.csv')
Mostpopular = Mostpopular.drop('Unnamed: 0', 1)

Leastpopular = pd.read_csv('C:/Users/abd/OneDrive/Bureau/PFE_FILES_FIN/LeastPopular.csv')
Leastpopular = Leastpopular.drop('Unnamed: 0', 1)

MostActive = pd.read_csv('C:/Users/abd/OneDrive/Bureau/PFE_FILES_FIN/MostActive.csv')
MostActive = MostActive.drop('Unnamed: 0', 1)

LeastActive = pd.read_csv('C:/Users/abd/OneDrive/Bureau/PFE_FILES_FIN/LeastActive.csv')
LeastActive = LeastActive.drop('Unnamed: 0', 1)

Similiarproducts = pd.read_excel('C:/Users/abd/OneDrive/Bureau/PFE_FILES_FIN/silimiarproducts.xlsx',header=None,index_col=False)
Similiarproducts.rename(columns={ Similiarproducts.columns[0]: "Similiar" }, inplace = True)

tasks = mongo.db.products
products = []
for field in tasks.find():
    products.append({'produitId': str(field['produitId']), 'Product_name': str(field['Product_name']),'Categories': str(field['Categories']),'subCategory': str(field['subCategory']),'description': str(field['description']),'Product_img': str(field['Product_img']),'Product_price': str(field['Product_price']),'baseColor': str(field['baseColor']),'gender': str(field['gender']),'vendeurId': str(field['vendeurId'])})
products = pd.DataFrame.from_records(products)

tasks = mongo.db.users
users = []
for field in tasks.find():
    users.append({'userId': str(field['userId']),'username': str(field['username']) ,'firstname': str(field['firstname']),'lastname': str(field['lastname']),'address': str(field['address']),'num_phone': str(field['num_phone']),'email': str(field['email']),'city': str(field['city']),'country': str(field['country']),'type': str(field['type'])})
users = pd.DataFrame.from_records(users)
    
tasks = mongo.db.metauser
metauser = []
for field in tasks.find():
    metauser.append({'userId': str(field['userId']),'produitId': str(field['produitId']) ,'rating': str(field['rating']),'like': str(field['like']),'panier': str(field['panier'])})
metauser = pd.DataFrame.from_records(metauser)

#like
likes = mongo.db.metauser
dflike = pd.DataFrame(list(likes.find()))
dflike = dflike.drop('_id', 1)
dflike = dflike.drop('rating', 1)
dflike = dflike[dflike.like != "-"]
dflike['like'] = dflike['like'].astype(np.int)
dflike['produitId'] = dflike['produitId'].astype(str)
dflike = pd.merge(dflike, products, how='left', on=['produitId'])

#rating
ratings = mongo.db.metauser
dfrating = pd.DataFrame(list(ratings.find()))
dfrating = dfrating.drop('_id', 1)
dfrating = dfrating.drop('like', 1)
dfrating = dfrating[dfrating.rating != "-"]
dfrating['rating'] = dfrating['rating'].astype(np.int)
dfrating['produitId'] = dfrating['produitId'].astype(str)
dfrating = pd.merge(dfrating, products, how='left', on=['produitId'])

@app.route("/api/search/<string:text>",methods=['GET'])
def search(text):
    text = str(text)
    print(text)
    NTLKsearch = open('C:/Users/abd/OneDrive/Bureau/PFE_FILES_FIN/test.txt', 'r')
    tfidf_vectorizer = TfidfVectorizer()
    count_matrix = tfidf_vectorizer.fit_transform(NTLKsearch)
    text = tfidf_vectorizer.transform([text])
    cosine_sim = cosine_similarity(text, count_matrix)

    Dataframe5 = pd.DataFrame(cosine_sim)
    Dataframe5 = Dataframe5.T
    Dataframe5.rename(columns={0: 'poncentage'}, inplace=True)
    Dataframe5["produitId"] = products.produitId.loc[Dataframe5.index == products.index]
    Dataframe5 = Dataframe5.sort_values('poncentage',ascending=False).head(200)
    Dataframe5['produitId'] = Dataframe5['produitId'].astype(str)
    Dataframe5 = Dataframe5.merge(products, on='produitId', how='left')
    rv = Dataframe5.to_json(orient='records')
    return rv

@app.route("/api/image/<string:img>", methods=['GET'])
def get_image(img):
    model = load_model('C:/Users/abd/OneDrive/Bureau/PFE_FILES_FIN/dataset/subweights.best.image_classifier.hdf5')
    test_image = image.load_img('C:/Users/abd/OneDrive/Bureau/Projet_FIN/src/assets/images/upload/'+img, target_size=(100, 100))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = model.predict(test_image)
    print(result)
    
    if result[0][0] == result.max(axis=1):
        prediction = 'Android'
    elif result[0][1] == result.max(axis=1):
        prediction = 'Apparel Set'
    elif result[0][2] == result.max(axis=1):
        prediction = 'Apple'
    elif result[0][3] == result.max(axis=1): 
        prediction = 'Bags Accessories'
    elif result[0][4] == result.max(axis=1):
        prediction = 'Bags Sporting'
    elif result[0][5] == result.max(axis=1):
        prediction = 'Bath Body'
    elif result[0][6] == result.max(axis=1): 
        prediction = 'Belts'
    elif result[0][7] == result.max(axis=1): 
        prediction = 'Bottomwear'
    elif result[0][8] == result.max(axis=1):
        prediction = 'Children'
    elif result[0][9] == result.max(axis=1):
        prediction = 'Classics'
    elif result[0][10] == result.max(axis=1): 
        prediction = 'Crime'
    elif result[0][11] == result.max(axis=1):
        prediction = 'Cufflinks'
    elif result[0][12] == result.max(axis=1):
        prediction = 'Dress'
    elif result[0][13] == result.max(axis=1): 
        prediction = 'Eyes'
    if result[0][14] == result.max(axis=1): 
        prediction = 'Eyewear Accessories'
    elif result[0][15] == result.max(axis=1):
        prediction = 'Eyewear Sporting'
    elif result[0][16] == result.max(axis=1):
        prediction = 'Families Relationship'
    elif result[0][17] == result.max(axis=1): 
        prediction = 'Fantasy'
    elif result[0][18] == result.max(axis=1):
        prediction = 'Flip Flops'
    elif result[0][19] == result.max(axis=1):
        prediction = 'Fragrance'
    elif result[0][20] == result.max(axis=1): 
        prediction = 'General'
    if result[0][21] == result.max(axis=1): 
        prediction = 'Gift'
    elif result[0][22] == result.max(axis=1):
        prediction = 'Gloves'
    elif result[0][23] == result.max(axis=1):
        prediction = 'Hair'
    elif result[0][24] == result.max(axis=1): 
        prediction = 'Headwear Accessories'
    elif result[0][25] == result.max(axis=1):
        prediction = 'Headwear Sporting'
    elif result[0][26] == result.max(axis=1):
        prediction = 'Honor'
    elif result[0][27] == result.max(axis=1): 
        prediction = 'Huawei'
    if result[0][28] == result.max(axis=1): 
        prediction = 'Innerwear'
    elif result[0][29] == result.max(axis=1):
        prediction = 'Jewellery'
    elif result[0][30] == result.max(axis=1):
        prediction = 'Lips'
    elif result[0][31] == result.max(axis=1): 
        prediction = 'Literature'
    elif result[0][32] == result.max(axis=1):
        prediction = 'Loungewear Nightwear'
    elif result[0][33] == result.max(axis=1):
        prediction = 'Makeup'
    elif result[0][34] == result.max(axis=1): 
        prediction = 'Nails'
    if result[0][35] == result.max(axis=1): 
        prediction = 'Nokia'
    elif result[0][36] == result.max(axis=1):
        prediction = 'Oneplus'
    elif result[0][37] == result.max(axis=1):
        prediction = 'Oppo'
    elif result[0][38] == result.max(axis=1): 
        prediction = 'Others'
    elif result[0][39] == result.max(axis=1):
        prediction = 'Redmi'
    elif result[0][40] == result.max(axis=1):
        prediction = 'Romance'
    elif result[0][41] == result.max(axis=1): 
        prediction = 'Samsung'
    if result[0][42] == result.max(axis=1): 
        prediction = 'Sandal'
    elif result[0][43] == result.max(axis=1):
        prediction = 'Saree'
    elif result[0][44] == result.max(axis=1):
        prediction = 'Scarves'
    elif result[0][45] == result.max(axis=1): 
        prediction = 'Science Fiction'
    elif result[0][46] == result.max(axis=1):
        prediction = 'Shoe Accessories'
    elif result[0][47] == result.max(axis=1):
        prediction = 'Shoes'
    elif result[0][48] == result.max(axis=1): 
        prediction = 'Short Stories'
    if result[0][49] == result.max(axis=1): 
        prediction = 'Skin Care'
    elif result[0][50] == result.max(axis=1):
        prediction = 'Socks'
    elif result[0][51] == result.max(axis=1):
        prediction = 'Sony'
    elif result[0][52] == result.max(axis=1): 
        prediction = 'Sport Equipement'
    elif result[0][53] == result.max(axis=1):
        prediction = 'Stoles'
    elif result[0][54] == result.max(axis=1):
        prediction = 'Ties'
    elif result[0][55] == result.max(axis=1): 
        prediction = 'Topwear'
    if result[0][56] == result.max(axis=1): 
        prediction = 'Vivo'
    elif result[0][57] == result.max(axis=1):
        prediction = 'Wallets'
    elif result[0][58] == result.max(axis=1):
        prediction = 'Watches Accessories'
    elif result[0][59] == result.max(axis=1): 
        prediction = 'Watches Sportings'
    elif result[0][60] == result.max(axis=1):
        prediction = 'Water Bottle'
    elif result[0][61] == result.max(axis=1):
        prediction = 'Young Adults'
    print(prediction)
    
    response = {'class' : prediction}
    return jsonify(response)

@app.route("/api/chatbot/<string:msg>", methods=['GET'])
def get_bot_response(msg):
    model = load_model('C:/Users/abd/OneDrive/Bureau/PFE_FILES_FIN/PFEchatbot_model.h5')
    intents = json.loads(open('C:/Users/abd/OneDrive/Bureau/PFE_FILES_FIN/PFE.json').read())
    words = pickle.load(open('C:/Users/abd/OneDrive/Bureau/PFE_FILES_FIN/PFEwords.pkl','rb'))
    classes = pickle.load(open('C:/Users/abd/OneDrive/Bureau/PFE_FILES_FIN/PFEclasses.pkl','rb'))
    def clean_up_sentence(sentence):
        # tokenize the question
        sentence_words = nltk.word_tokenize(sentence)
        # lemmatize and lower for each word
        sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
        return sentence_words

   # return bag of words: 1 for each word that exists in the bag of words and in the question (sentence) and 0 for the other words
    def bow(sentence, words):
        # tokenize lemmaztize and lower the question
        sentence_words = clean_up_sentence(sentence)
        # initialize our bag of words by 0 for all words
        bag = [0]*len(words) 
        for s in sentence_words:
            for i,w in enumerate(words): # eg [(0, «manger»), (1, «dormir»)] so (i, «w»)
                if w == s: 
                    # assign 1 if the current word is in the question
                    bag[i] = 1
        return(np.array(bag))

    def predict_class(sentence, model):
        # call tha matrix of bag of words
        p = bow(sentence, words)
        res = model.predict(np.array([p]))[0] # predection of the class belong this sentence
        #filter predictions below a threshold
        ERROR_THRESHOLD = 0.25
        results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD] # eg if we have 2 classes : {[1,("class 0", 0.5)], [2,("class 1", 0.90)]} so (i, r)
        # sort by strength of probability
        results.sort(key=lambda x: x[1], reverse=True) # output the result of predections for the classes
        return_list = []
        for r in results:
            return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    
        return return_list

    def getResponse(ints, intents_json):
        tag = ints[0]['intent'] # take the class with the highest prediction
        list_of_intents = intents_json['intents'] # list of all classes
        for i in list_of_intents:
            if(i['tag']== tag):
                result = random.choice(i['responses']) # choose a random response from the class that has the greatest prediction
                break
        return result

    def chatbot_response(text):
        ints = predict_class(text, model)
        print(ints)
        res = getResponse(ints, intents)
        return res

    response = {'response' : chatbot_response(msg)}
    return jsonify(response)

@app.route('/api/tasks', methods=['GET'])
def get_all_tasks():
    Populars = Mostpopular.head(20)
    rv = Populars.to_json(orient='records')
    return rv

@app.route('/api/taskscat/<string:catg>', methods=['GET'])
def get_all_taskscat(catg):
    
    rv = Mostpopular.to_json(orient='records')
    # Transform json input to python objects
    input_dict = json.loads(rv)
    # Filter python objects with list comprehensions
    output_dict = [x for x in input_dict if x['Categories'] == catg]
    # Transform python object back into json
    output_json = json.dumps(output_dict)
    
    return output_json

@app.route('/api/taskssubcat/<string:catg>/<string:subCat>', methods=['GET'])
def get_all_taskssubcat(catg,subCat):
    Popularscat = Mostpopular.head(2000)
    rv = Popularscat.to_json(orient='records')
    # Transform json input to python objects
    input_dict = json.loads(rv)
    # Filter python objects with list comprehensions
    output_dict = [x for x in input_dict if x['Categories'] == catg and x['subCategory'] == subCat ]
    # Transform python object back into json
    output_json = json.dumps(output_dict)
    return output_json

@app.route('/api/tasks/<id>', methods=['GET'])
def get_all_tasksuser(id):
    id = int(id)
    Mostpopularuser = Popular.loc[Popular['vendeurId'] == id]
    Mostpopularuser = Mostpopularuser.head(10)
    rv = Mostpopularuser.to_json(orient='records')
    return rv

@app.route('/api/leasttasks', methods=['GET'])
def get_Least_popular_products():
    rv = Leastpopular.to_json(orient='records')
    return rv

@app.route('/api/leasttasks/<id>', methods=['GET'])
def get_Least_popular_productsuser(id):
    id =  int(id)
    Leastpopularuser = Popular.loc[Popular['vendeurId'] == id]
    Leastpopularuser = Leastpopularuser.tail(10)
    rv = Leastpopularuser.to_json(orient='records')
    return rv

@app.route('/api/activeusers', methods=['GET'])
def get_active_users():
    rv = MostActive.to_json(orient='records')
    return rv

@app.route('/api/leastactiveusers', methods=['GET'])
def get_least_active_users():
    rv = LeastActive.to_json(orient='records')
    return rv

@app.route('/api/similiar/<int:id>')
def Similiar(id):
    id = str(id)
    list = products.index[products.produitId == id].tolist()
    id = int(list[0])
    return Similiarproducts['Similiar'][id]

@app.route('/api/maylike/<int:id>')
def getalltasks(id):
    
    if(RatingsLikes.loc[RatingsLikes['userId'] == id].empty == False):
        RatingsLikesdataframe = pd.read_json (RatingsLikes.loc[RatingsLikes['userId'] == id].iloc[0]['Maylike'])
        RatingsLikesdataframe = RatingsLikesdataframe['produitId'].astype(str)
        RatingsLikesdataframe = pd.merge(RatingsLikesdataframe, products, how='left', on=['produitId'])
        RatingsLikesdataframe = shuffle(RatingsLikesdataframe)
        RatingsLikesjson = RatingsLikesdataframe.to_json(orient='records')
    else:
        Populars = Mostpopular.head(20)
        RatingsLikesjson = Populars.to_json(orient='records')
    
    return RatingsLikesjson

@app.route('/api/maylike/<int:id>/<string:catg>')
def getmaylikecat(id,catg):
    
    if(RatingsLikes.loc[RatingsLikes['userId'] == id].empty == False):
        RatingsLikescatdataframe = pd.read_json(RatingsLikes.loc[RatingsLikes['userId'] == id].iloc[0]['Maylike'])
        RatingsLikescatdataframe = RatingsLikescatdataframe['produitId'].astype(str)
        RatingsLikescatdataframe = pd.merge(RatingsLikescatdataframe, products, how='left', on=['produitId'])
        RatingsLikescatdataframe = shuffle(RatingsLikescatdataframe)
        resultatscat = RatingsLikescatdataframe.loc[(RatingsLikescatdataframe['Categories'] == catg)]
        RatingsLikescatjson = resultatscat.to_json(orient='records')
        if(len(RatingsLikescatjson) == 2):
            RatingsLikescatjson = RatingsLikescatdataframe.to_json(orient='records')
    else:
        rv = Mostpopular.to_json(orient='records')
        # Transform json input to python objects
        input_dict = json.loads(rv)
        # Filter python objects with list comprehensions
        output_dict = [x for x in input_dict if x['Categories'] == catg]
        # Transform python object back into json
        RatingsLikescatjson = json.dumps(output_dict)
    
    return RatingsLikescatjson


@app.route('/api/maylike/<int:id>/<string:catg>/<string:subCat>')
def getmaylikesubcat(id,catg,subCat):
    
    if(RatingsLikes.loc[RatingsLikes['userId'] == id].empty == False):
        RatingsLikessubcatdataframe = pd.read_json (RatingsLikes.loc[RatingsLikes['userId'] == id].iloc[0]['Maylike'])
        RatingsLikessubcatdataframe = RatingsLikessubcatdataframe['produitId'].astype(str)
        RatingsLikessubcatdataframe = pd.merge(RatingsLikessubcatdataframe, products, how='left', on=['produitId'])
        RatingsLikessubcatdataframe = shuffle(RatingsLikessubcatdataframe)
        RatingsLikessubcatdataframecat = RatingsLikessubcatdataframe.loc[(RatingsLikessubcatdataframe['Categories'] == catg)]
        resultatssub = RatingsLikessubcatdataframecat.loc[(RatingsLikessubcatdataframe['subCategory'] == subCat)]    
        RatingsLikessubcatjson = resultatssub.to_json(orient='records')
        
        if(len(RatingsLikessubcatjson) == 2):
            RatingsLikessubcatjson = RatingsLikessubcatdataframecat.to_json(orient='records')    
        if(len(RatingsLikessubcatjson) == 2):
            RatingsLikessubcatjson = RatingsLikessubcatdataframe.to_json(orient='records')           
    else:
        Popularscat = Mostpopular.head(2000)
        rv = Popularscat.to_json(orient='records')
        # Transform json input to python objects
        input_dict = json.loads(rv)
        # Filter python objects with list comprehensions
        output_dict = [x for x in input_dict if x['Categories'] == catg and x['subCategory'] == subCat ]
        # Transform python object back into json
        RatingsLikessubcatjson = json.dumps(output_dict)
        
    return RatingsLikessubcatjson

@app.route('/api/all', methods=['GET'])
def get_all():
    tasks = mongo.db.products 
    result = []
    for field in tasks.find():
        result.append({'produitId': str(field['produitId']), 'Product_name': str(field['Product_name']),'Categories': str(field['Categories']),'subCategory': str(field['subCategory']),'description': str(field['description']),'Product_img': str(field['Product_img']),'Product_price': str(field['Product_price']),'baseColor': str(field['baseColor']),'gender': str(field['gender']),'vendeurId': str(field['vendeurId'])})
    return jsonify(result)

@app.route('/api/produit/<id>', methods=['GET'])
def get_produit(id):
    tasks = mongo.db.products 
    result = []
    id = int(id)
    for field in tasks.find({'produitId':id}):
        result.append({'produitId': str(field['produitId']), 'Product_name': str(field['Product_name']),'Categories': str(field['Categories']),'subCategory': str(field['subCategory']),'description': str(field['description']),'Product_img': str(field['Product_img']),'Product_price': str(field['Product_price']),'baseColor': str(field['baseColor']),'gender': str(field['gender']),'vendeurId': str(field['vendeurId'])})
    return jsonify(result)

@app.route('/api/all/<id>', methods=['GET'])
def get_alluser(id):
    tasks = mongo.db.products 
    result = []
    id = int(id)
    for field in tasks.find({"vendeurId" : id}):
        result.append({'produitId': str(field['produitId']), 'Product_name': str(field['Product_name']),'Categories': str(field['Categories']),'subCategory': str(field['subCategory']),'description': str(field['description']),'Product_img': str(field['Product_img']),'Product_price': str(field['Product_price']),'baseColor': str(field['baseColor']),'gender': str(field['gender'])})
    return jsonify(result)

@app.route('/api/metausers', methods=['GET'])
def get_metausers():
    tasks = mongo.db.metauser
    metauser = []
    for field in tasks.find():
        metauser.append({'userId': str(field['userId']),'produitId': str(field['produitId']) ,'rating': str(field['rating']),'like': str(field['like']),'panier': str(field['panier'])})
    metauser = pd.DataFrame.from_records(metauser)
    users_metausers = metauser.merge(users, on='userId', how='left')
    users_metausers = users_metausers.merge(products, on='produitId', how='left')
    users_metausers =users_metausers[["userId","username","produitId","Product_name","rating","like","panier"]]
    users_metausers = users_metausers.to_json(orient='records')
    return users_metausers

@app.route('/api/metauser/<id>', methods=['GET'])
def get_metauser(id):
    id =  int(id)
    tasks = mongo.db.metauser
    metauser = []
    for field in tasks.find():
        metauser.append({'userId': str(field['userId']),'produitId': str(field['produitId']) ,'rating': str(field['rating']),'like': str(field['like']),'panier': str(field['panier'])})
    metauser = pd.DataFrame.from_records(metauser)
    users_metausers = metauser.merge(users, on='userId', how='left')
    id =  str(id)
    users_metausers = users_metausers.loc[users_metausers['userId'] == id]
    users_metausers = users_metausers.merge(products, on='produitId', how='left')
    users_metausers =users_metausers[["userId","username","produitId","Product_name","rating","like","panier"]]
    users_metausers = users_metausers.to_json(orient='records')
    return users_metausers

@app.route('/api/users', methods=['GET'])
def get_users():
    tasks = mongo.db.users 
    result = []
    for field in tasks.find():
        result.append({'_id': str(field['_id']),'userId': str(field['userId']),'username': str(field['username']) ,'password': str(field['password']),'firstname': str(field['firstname']),'lastname': str(field['lastname']),'address': str(field['address']),'num_phone': str(field['num_phone']),'email': str(field['email']),'city': str(field['city']),'country': str(field['country']),'type': str(field['type'])})
    return jsonify(result)

@app.route('/api/user/<id>', methods=['GET'])
def getUser(id):
    user = mongo.db.users 
    result = []
    print(type(id))
    id =  int(id)
    new_user = user.find_one({'userId': id})
    result = {'_id': str(new_user['_id']),'userId': str(new_user['userId']),'username': str(new_user['username']) ,'password': str(new_user['password']),'firstname': str(new_user['firstname']),'lastname': str(new_user['lastname']),'address': str(new_user['address']),'num_phone': str(new_user['num_phone']),'email': str(new_user['email']),'city': str(new_user['city']),'country': str(new_user['country']),'type': str(new_user['type']),'image': str(new_user['image'])}
    print(result)
    return jsonify(result)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/upload_file',  methods=["GET", "POST"])
def upload_file():
    print(request.files)
    # check if the post request has the file part
    if 'file' not in request.files:
        print('no file in request')
        return""
    file = request.files['file']
    if file.filename == '':
        print('no selected file')
        return""
    if file and allowed_file(file.filename):
        print("hello")
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return ""
    print("end")
    return""
    
    return "uploaded"  
 
@app.route('/api/metausers/panierdelete/<produitid>/<userid>', methods=['DELETE'])
def delete_panier(produitid,userid):
    tasks = mongo.db.metauser
    produitid =  int(produitid)
    userid =  int(userid)
    tasks.delete_one({'produitId': produitid , 'userId':userid,'panier':1})
        
    result = {'message':'Done'}
    return jsonify(result)

@app.route('/api/metausers/panier', methods=['POST'])
def add_panier():
    tasks = mongo.db.metauser
    produitId = request.get_json()['produitId']
    userId = request.get_json()['userId']
    task = tasks.find({'produitId': int(produitId) , 'userId':int(userId), 'panier':1})
    if (task.count() == 0):
        tasks.insert_one({'produitId': int(produitId) , 'userId':int(userId), 'panier':1 , 'rating':'-' , 'like':'-' , 'timestamp':datetime.now() })
    else:
        print("produit already in panier")
        
    result = {'message':'Done'}
    return jsonify(result)

@app.route('/api/metausers/like', methods=['POST'])
def add_like():
    tasks = mongo.db.metauser
    produitId = request.get_json()['produitId']
    userId = request.get_json()['userId']
    task = tasks.find({'produitId': int(produitId) , 'userId':int(userId), 'rating':'-' , 'panier':'-'})
    if (task.count() == 0):
        tasks.insert_one({'produitId': int(produitId) , 'userId':int(userId), 'like':1 , 'rating':'-' , 'panier':'-' , 'timestamp':datetime.now() })
    else:
        tasks.delete_one({'produitId': int(produitId) , 'userId':int(userId),'rating':'-' , 'panier':'-'})
         
    result = {'message':'Done'}
    return jsonify(result)

@app.route('/api/metausers/rating', methods=['POST'])
def add_rating():
    tasks = mongo.db.metauser
    produitId = request.get_json()['produitId']
    userId = request.get_json()['userId']
    rating = request.get_json()['rating']
    
    task = tasks.find({'produitId': int(produitId) , 'userId':int(userId), 'like':'-' , 'panier':'-'})
    if (task.count() == 0):
        tasks.insert_one({'produitId': int(produitId) , 'userId':int(userId), 'rating':int(rating) , 'like':'-' , 'panier':'-' , 'timestamp':datetime.now() })
    else:
        tasks.find_one_and_update({'produitId': int(produitId) , 'userId':int(userId), 'like':'-' , 'panier':'-'},{'$set': {'produitId': int(produitId) , 'userId':int(userId), 'rating':int(rating) , 'like':'-' , 'panier':'-' , 'timestamp':datetime.now() }}, upsert=False)
        
    result = {'message':'Done'}
    return jsonify(result)

@app.route('/api/task', methods=['POST'])
def add_task():
    tasks = mongo.db.products
    produitId = request.get_json()['produitId']
    Product_name = request.get_json()['Product_name']
    description = request.get_json()['description']
    Categories = request.get_json()['Categories']
    Product_price = request.get_json()['Product_price']
    subCategory = request.get_json()['subCategory']
    gender = request.get_json()['gender']
    baseColor = request.get_json()['baseColor']
    Product_img = request.get_json()['Product_img']
    vendeurId = request.get_json()['vendeurId']
    vendeurId =  int(vendeurId)
    produitId = int(produitId)
    Product_price = int(Product_price)
    task_id = tasks.insert({'produitId': produitId , 'Product_name':Product_name, 'description':description , 'Categories':Categories , 'Product_price':Product_price , 'subCategory':subCategory , 'gender':gender , 'baseColor':baseColor , 'Product_img':Product_img , 'vendeurId':vendeurId })
    
    new_task = tasks.find_one({'_id': task_id})
    print(new_task)
    result = {'produitId': new_task['produitId'],'Product_name': new_task['Product_name'],'description': new_task['description'],'Categories': new_task['Categories'],'Product_price': new_task['Product_price'],'subCategory': new_task['subCategory'],'gender': new_task['gender'],'baseColor': new_task['baseColor'],'Product_img': new_task['Product_img'],'vendeurId':new_task['vendeurId']}
    print(result)
    return jsonify(result)

@app.route('/api/users', methods=['POST'])
def addUser():
    tasks = mongo.db.users
    Email = request.get_json()['email']
    Password = request.get_json()['password']
    Username = request.get_json()['username']
    Address = request.get_json()['address']
    Num_phone = request.get_json()['num_phone']
    City = request.get_json()['city']
    Country = request.get_json()['country']
    Firstname = request.get_json()['firstname']
    Lastname = request.get_json()['lastname']
    Type = request.get_json()['type']
    image = request.get_json()['image']
    
    task_id = tasks.insert({'userId': int(tasks.count()+1),'email': Email,'password':Password,'username':Username, 'address':Address , 'num_phone':int(Num_phone) , 'city':City , 'country':Country,'firstname':Firstname,'lastname':Lastname,'type':Type,'image':image})
    
    new_task = tasks.find_one({'_id': task_id})
    print(new_task)
    result = {'userId': new_task[tasks.count()+1],'email': new_task['Email'],'password': new_task['Password'],'username': new_task['Username'],'address': new_task['Adress'],'num_phone': new_task['Num_phone'],'city': new_task['City'],'country': new_task['Country'],'firstname': new_task['Firstname'],'lastname': new_task['Lastname'],'type': new_task['Type'],'image': new_task['image']}
    print(result)
    return jsonify(result)

@app.route('/api/<string:catg>', methods=['GET'])
def get_categorie(catg):
    tasks = mongo.db.products 
    result = []
    for field in tasks.find({"Categories" : catg}):
        result.append({'_id': str(field['_id']), 'produitId':field['produitId'],'Product_name': field['Product_name'] ,'Categories': str(field['Categories']),'subCategory': str(field['subCategory']),'description': str(field['description']),'Product_img': str(field['Product_img']),'Product_price': str(field['Product_price']),'baseColor': str(field['baseColor']),'gender': str(field['gender'])})
    return jsonify(result)

@app.route('/api/<string:catg>/<string:subCat>', methods=['GET'])
def get_subCategory(catg ,subCat):
    tasks = mongo.db.products 
    result = []
    for field in tasks.find({"Categories" : catg ,"subCategory" : subCat }):
        result.append({'_id': str(field['_id']), 'produitId':field['produitId'],'Product_name': field['Product_name'] ,'Categories': str(field['Categories']),'subCategory': str(field['subCategory']),'description': str(field['description']),'Product_img': str(field['Product_img']),'Product_price': str(field['Product_price']),'baseColor': str(field['baseColor']),'gender': str(field['gender'])})
    return jsonify(result)

@app.route('/api/updateuser/<id>', methods=['PUT'])
def updateuser(id):
    tasks = mongo.db.users
    
    username = request.get_json()['username']
    firstname = request.get_json()['firstname']
    lastname = request.get_json()['lastname']
    email = request.get_json()['email']
    password = request.get_json()['password']
    address = request.get_json()['address']
    country = request.get_json()['country']
    city = request.get_json()['city']
    num_phone = request.get_json()['num_phone']
    image = request.get_json()['image']
    
    print(type(id))
    id =  int(id)
    
    tasks.find_one_and_update({'userId': id}, {'$set': {'email': email,'password':password,'username':username, 'address':address , 'num_phone':int(num_phone) , 'city':city , 'country':country,'firstname':firstname,'lastname':lastname,'image':image}}, upsert=False)
    new_task = tasks.find_one({'userId': id})
    
    result = {'userId': new_task['userId'],'email': new_task['email'],'password': new_task['password'],'username': new_task['username'],'address': new_task['address'],'num_phone': new_task['num_phone'],'city': new_task['city'],'country': new_task['country'],'firstname': new_task['firstname'],'lastname': new_task['lastname'],'image': new_task['image']}
    print(result)
    return jsonify({'result': result})

@app.route('/api/users/<id>', methods=['DELETE'])
def deleteUser(id): 
    id =  int(id)
    
    response = mongo.db.users.delete_one({'userId': id})
    mongo.db.metauser.delete_many({'userId': id})

    if response.deleted_count == 1:
        result = {'message': 'record deleted'}
    else: 
        result = {'message': 'no record found'}
    
    return jsonify({'result': result})

@app.route('/api/taskupdate/<id>', methods=['PUT'])
def update_task(id):
    tasks = mongo.db.products
    
    Product_name = request.get_json()['Product_name']
    description = request.get_json()['description']
    Categories = request.get_json()['Categories']
    Product_price = request.get_json()['Product_price']
    gender = request.get_json()['gender']
    baseColor = request.get_json()['baseColor']
    Product_img = request.get_json()['Product_img']
    Product_price = int(Product_price)
    print(gender)
    id = int(id)
    tasks.find_one_and_update({'produitId': id}, {'$set': {'Product_name': Product_name,'description':description,'Categories':Categories,'Product_price':Product_price,'gender':gender,'baseColor':baseColor,'Product_img':Product_img}}, upsert=False)
    new_task = tasks.find_one({'produitId': id})

    result = {'Product_name': new_task['Product_name'],'description': new_task['description'],'Categories': new_task['Categories'],'Product_price': new_task['Product_price'],'gender': new_task['gender'],'baseColor': new_task['baseColor'],'Product_img': new_task['Product_img']}
    print(result)
    return jsonify({'result': result})

@app.route('/api/task/<id>', methods=['DELETE'])
def delete_task(id):
    id =  int(id)
    response = mongo.db.products.delete_one({'produitId': id})
    mongo.db.metauser.delete_many({'produitId': id})
    if response.deleted_count == 1:
        result = {'message': 'record deleted'}
        print(result)
    else: 
        result = {'message': 'no record found'}
        print(result)
    
    return jsonify({'result': result})

@app.route('/api/info', methods=['GET'])
def informations():
    tasks = mongo.db.products
    info = []
    #count Categories
    Accessories = tasks.find( {"Categories" : "Accessories"} ).count()
    Apparel = tasks.find( {"Categories" : "Apparel"} ).count()
    Books = tasks.find( {"Categories" : "Books"} ).count()
    Footwear = tasks.find( {"Categories" : "Footwear"} ).count()
    Personal_Care = tasks.find( {"Categories" : "Personal Care"} ).count()
    Phones = tasks.find( {"Categories" : "Phones"} ).count()
    Sporting = tasks.find( {"Categories" : "Sporting Goods"} ).count()
    #count subcategories
    belts = tasks.find( {"subCategory" : "belts"} ).count()
    jewellery = tasks.find( {"subCategory" : "jewellery"} ).count()
    eyewear_Accessories = tasks.find( {"Categories" : "Accessories","subCategory" : "eyewear"} ).count()
    bags_Accessories = tasks.find( {"Categories" : "Accessories","subCategory" : "bags"} ).count()
    headwear_Accessories = tasks.find( {"Categories" : "Accessories","subCategory" : "headwear"} ).count()
    wallets = tasks.find( {"subCategory" : "wallets"} ).count()
    cufflinks = tasks.find( {"subCategory" : "cufflinks"} ).count()
    gloves = tasks.find( {"subCategory" : "gloves"} ).count()
    watches_Accessories = tasks.find( {"Categories" : "Accessories","subCategory" : "watches"} ).count()
    gift = tasks.find( {"subCategory" : "accessory gift set"} ).count()
    innerwear = tasks.find( {"subCategory" : "innerwear"} ).count()
    scarves = tasks.find( {"subCategory" : "scarves"} ).count()
    bottomwear = tasks.find( {"subCategory" : "bottomwear"} ).count()
    topwear = tasks.find( {"subCategory" : "topwear"} ).count()
    apparel_set = tasks.find( {"subCategory" : "apparel set"} ).count()
    dress = tasks.find( {"subCategory" : "dress"} ).count()
    ties = tasks.find( {"subCategory" : "ties"} ).count()
    stoles = tasks.find( {"subCategory" : "stoles"} ).count()
    loungewear_nightwear = tasks.find( {"subCategory" : "loungewear and nightwear"} ).count()
    saree = tasks.find( {"subCategory" : "saree"} ).count()
    general = tasks.find( {"subCategory" : "general"} ).count()
    others = tasks.find( {"subCategory" : "others"} ).count()
    literature = tasks.find( {"subCategory" : "literature"} ).count()
    families_relationship = tasks.find( {"subCategory" : "families & relationship"} ).count()
    fantasy = tasks.find( {"subCategory" : "fantasy"} ).count()
    children = tasks.find( {"subCategory" : "children"} ).count()
    classics = tasks.find( {"subCategory" : "classics"} ).count()
    crime = tasks.find( {"subCategory" : "crime"} ).count()
    short_stories = tasks.find( {"subCategory" : "short stories"} ).count()
    young_adults = tasks.find( {"subCategory" : "young adults"} ).count()
    romance = tasks.find( {"subCategory" : "romance"} ).count()
    science_fiction = tasks.find( {"subCategory" : "science fiction"} ).count()
    socks = tasks.find( {"subCategory" : "socks"} ).count()
    shoes = tasks.find( {"subCategory" : "shoes"} ).count()
    flip_flops = tasks.find( {"subCategory" : "flip flops"} ).count()
    sandal = tasks.find( {"subCategory" : "sandal"} ).count()
    shoe_accessories = tasks.find( {"subCategory" : "shoe accessories"} ).count()
    fragrance = tasks.find( {"subCategory" : "fragrance"} ).count()
    lips = tasks.find( {"subCategory" : "lips"} ).count()
    eyes = tasks.find( {"subCategory" : "eyes"} ).count()
    hair = tasks.find( {"subCategory" : "hair"} ).count()
    makeup = tasks.find( {"subCategory" : "makeup"} ).count()
    bath_body = tasks.find( {"subCategory" : "bath and body"} ).count()
    skin_care = tasks.find( {"subCategory" : "skin care"} ).count()
    nails = tasks.find( {"subCategory" : "nails"} ).count()
    android = tasks.find( {"subCategory" : "android"} ).count()
    honor = tasks.find( {"subCategory" : "honor"} ).count()
    vivo = tasks.find( {"subCategory" : "vivo"} ).count()
    sony = tasks.find( {"subCategory" : "sony"} ).count()
    oppo = tasks.find( {"subCategory" : "oppo"} ).count()
    redmi = tasks.find( {"subCategory" : "redmi"} ).count()
    nokia = tasks.find( {"subCategory" : "nokia"} ).count()
    oneplus = tasks.find( {"subCategory" : "oneplus"} ).count()
    apple = tasks.find( {"subCategory" : "apple"} ).count()
    samsung = tasks.find( {"subCategory" : "samsung"} ).count()
    huawei = tasks.find( {"subCategory" : "huawei"} ).count()
    bags_Sporting = tasks.find( {"Categories" : "Sporting Goods","subCategory" : "bags"} ).count()
    headwear_Sporting = tasks.find( {"Categories" : "Sporting Goods","subCategory" : "headwear"} ).count()
    water_bottle = tasks.find( {"subCategory" : "water bottle"} ).count()
    eyewear_Sporting = tasks.find( {"Categories" : "Sporting Goods","subCategory" : "eyewear"} ).count()
    watches_Sporting = tasks.find( {"Categories" : "Sporting Goods","subCategory" : "watches"} ).count()
    sports_equipment = tasks.find( {"subCategory" : "sports equipment"} ).count()
     
    info.append({'accessoriescount':Accessories,'apparelcount':Apparel,'bookscount':Books,'footwearcount':Footwear,'personal_Carecount':Personal_Care,'phonescount':Phones,'sportcount':Sporting})  
    info.append({'beltscount':belts,'jewellerycount':jewellery,'eyewear_Accessoriescount':eyewear_Accessories,'bags_Accessoriescount':bags_Accessories,'headwear_Accessoriescount':headwear_Accessories,'walletscount':wallets,'cufflinkscount':cufflinks,'glovescount':gloves,'watches_Accessoriescount':watches_Accessories,'giftcount':gift})
    info.append({'innerwearcount':innerwear,'scarvescount':scarves,'bottomwearcount':bottomwear,'topwearcount':topwear,'apparel_setcount':apparel_set,'dresscount':dress,'tiescount':ties,'stolescount':stoles,'loungewear_nightwearcount':loungewear_nightwear,'sareecount':saree})
    info.append({'generalcount':general,'otherscount':others,'literaturecount':literature,'families_relationshipcount':families_relationship,'fantasycount':fantasy,'childrencount':children,'classicscount':classics,'crimecount':crime,'short_storiescount':short_stories,'young_adultscount':young_adults,'romancecount':romance,'science_fictioncount':science_fiction})
    info.append({'sockscount':socks,'shoescount':shoes,'flip_flopscount':flip_flops,'sandalcount':sandal,'shoe_accessoriescount':shoe_accessories})
    info.append({'fragrancecount':fragrance,'lipscount':lips,'eyecount':eyes,'haircount':hair,'makeupcount':makeup,'bath_bodycount':bath_body,'skin_carecount':skin_care,'nailscount':nails})
    info.append({'androidcount':android,'honorcount':honor,'vivocount':vivo,'sonycount':sony,'oppocount':oppo,'redmicount':redmi,'nokiacount':nokia,'onepluscount':oneplus,'applecount':apple,'samsungcount':samsung,'huaweicount':huawei})
    info.append({'bags_Sportingcount':bags_Sporting,'headwear_Sportingcount':headwear_Sporting,'water_bottlecount':water_bottle,'eyewear_Sportingcount':eyewear_Sporting,'watches_Sportingcount':watches_Sporting,'sports_equipmentcount':sports_equipment})
    
    #Most rated by category
    countrating = dfrating.groupby("Categories", as_index=False).count()
    meanrating = dfrating.groupby("Categories", as_index=False).mean()
    Mostrated = pd.merge(countrating, meanrating, how='right', on=['Categories'])
    Mostrated["Count"] = Mostrated["rating_x"]
    Mostrated["Mean"] = Mostrated["rating_y"]
    Mostrated = Mostrated[['Categories',"Count","Mean"]]
    Mostrated['Count'] = Mostrated['Count'].astype(str)
    Mostrated['Mean'] = Mostrated['Mean'].astype(str)
    
    #Most liked by category
    countlike = dflike.groupby("Categories", as_index=False).count()
    countlike["Countlike"] = countlike["like"]
    MostLiked = countlike[['Categories',"Countlike"]]
    MostLiked['Countlike'] = MostLiked['Countlike'].astype(str)
    
    info.append({'accessoriestotalrating':Mostrated.loc[0,'Count'],'accessoriesrated':Mostrated.loc[0,'Mean'],'accessoriestotalliking':MostLiked.loc[0,'Countlike']})
    info.append({'appareltotalrating':Mostrated.loc[1,'Count'],'apparelrated':Mostrated.loc[1,'Mean'],'appareltotalliking':MostLiked.loc[1,'Countlike']})
    info.append({'bookstotalrating':Mostrated.loc[2,'Count'],'booksrated':Mostrated.loc[2,'Mean'],'bookstotalliking':MostLiked.loc[2,'Countlike']})
    info.append({'footweartotalrating':Mostrated.loc[3,'Count'],'footwearrated':Mostrated.loc[3,'Mean'],'footweartotalliking':MostLiked.loc[3,'Countlike']})
    info.append({'personal_Caretotalrating':Mostrated.loc[4,'Count'],'personal_Carerated':Mostrated.loc[4,'Mean'],'personal_Caretotalliking':MostLiked.loc[4,'Countlike']})
    info.append({'phonestotalrating':Mostrated.loc[5,'Count'],'phonesrated':Mostrated.loc[5,'Mean'],'phonestotalliking':MostLiked.loc[5,'Countlike']})
    info.append({'sporttotalrating':Mostrated.loc[6,'Count'],'sportrated':Mostrated.loc[6,'Mean'],'sporttotalliking':MostLiked.loc[6,'Countlike']})
    
    #Most rated by subcategory
    subcountrating = dfrating.groupby(['Categories','subCategory'], as_index=False).count()
    submeanrating = dfrating.groupby(['Categories','subCategory'], as_index=False).mean()
    subMostrated = pd.merge(subcountrating, submeanrating, how='left', on=['Categories','subCategory'])
    subMostrated["Count"] = subMostrated["rating_x"]
    subMostrated["Mean"] = subMostrated["rating_y"]
    subMostrated = subMostrated[['Categories','subCategory',"Count","Mean"]]
    subMostrated['Count'] = subMostrated['Count'].astype(str)
    subMostrated['Mean'] = subMostrated['Mean'].astype(str)
    
    #Most liked by subcategory
    subcountlike = dflike.groupby(['Categories','subCategory'], as_index=False).count()
    subcountlike["Countlike"] = subcountlike["like"]
    subMostLiked = subcountlike[['Categories','subCategory',"Countlike"]]
    subMostLiked['Countlike'] = subMostLiked['Countlike'].astype(str)
    
    info.append({'gifttotalrating':subMostrated.loc[0,'Count'],'giftsrated':subMostrated.loc[0,'Mean'],'giftstotalliking':subMostLiked.loc[0,'Countlike']})
    info.append({'bags_Accessoriestotalrating':subMostrated.loc[1,'Count'],'bags_Accessoriesrated':subMostrated.loc[1,'Mean'],'bags_Accessoriestotalliking':subMostLiked.loc[1,'Countlike']})
    info.append({'beltstotalrating':subMostrated.loc[2,'Count'],'beltsrated':subMostrated.loc[2,'Mean'],'beltstotalliking':subMostLiked.loc[2,'Countlike']})
    info.append({'cufflinkstotalrating':subMostrated.loc[3,'Count'],'cufflinksrated':subMostrated.loc[3,'Mean'],'cufflinkstotalliking':subMostLiked.loc[3,'Countlike']})
    info.append({'eyewear_Accessoriestotalrating':subMostrated.loc[4,'Count'],'eyewear_Accessoriesrated':subMostrated.loc[4,'Mean'],'eyewear_Accessoriestotalliking':subMostLiked.loc[4,'Countlike']})
    info.append({'glovestotalrating':subMostrated.loc[5,'Count'],'glovesrated':subMostrated.loc[5,'Mean'],'glovestotalliking':subMostLiked.loc[5,'Countlike']})
    info.append({'headwear_Accessoriestotalrating':subMostrated.loc[6,'Count'],'headwear_Accessoriesrated':subMostrated.loc[6,'Mean'],'headwear_Accessoriestotalliking':subMostLiked.loc[6,'Countlike']})
    info.append({'jewellerytotalrating':subMostrated.loc[7,'Count'],'jewelleryrated':subMostrated.loc[7,'Mean'],'jewellerytotalliking':subMostLiked.loc[7,'Countlike']})
    info.append({'walletstotalrating':subMostrated.loc[8,'Count'],'walletsrated':subMostrated.loc[8,'Mean'],'walletstotalliking':subMostLiked.loc[8,'Countlike']})
    info.append({'watches_Accessoriestotalrating':subMostrated.loc[9,'Count'],'watches_Accessoriesrated':subMostrated.loc[9,'Mean'],'watches_Accessoriestotalliking':subMostLiked.loc[9,'Countlike']})
    info.append({'apparel_settotalrating':subMostrated.loc[10,'Count'],'apparel_setrated':subMostrated.loc[10,'Mean'],'apparel_settotalliking':subMostLiked.loc[10,'Countlike']})
    info.append({'bottomweatotalrrating':subMostrated.loc[11,'Count'],'bottomwearrated':subMostrated.loc[11,'Mean'],'bottomweartotalliking':subMostLiked.loc[11,'Countlike']})
    info.append({'dresstotalrating':subMostrated.loc[12,'Count'],'dressrated':subMostrated.loc[12,'Mean'],'dresstotalliking':subMostLiked.loc[12,'Countlike']})
    info.append({'innerweartotalrating':subMostrated.loc[13,'Count'],'innerwearrated':subMostrated.loc[13,'Mean'],'innerweartotalliking':subMostLiked.loc[13,'Countlike']})
    info.append({'loungewear_nightweartotalrating':subMostrated.loc[14,'Count'],'loungewear_nightwearrated':subMostrated.loc[14,'Mean'],'loungewear_nightweartotalliking':subMostLiked.loc[14,'Countlike']})
    info.append({'sareetotalrating':subMostrated.loc[15,'Count'],'sareerated':subMostrated.loc[15,'Mean'],'sareetotalliking':subMostLiked.loc[15,'Countlike']})
    info.append({'scarvestotalrating':subMostrated.loc[16,'Count'],'scarvesrated':subMostrated.loc[16,'Mean'],'scarvestotalliking':subMostLiked.loc[16,'Countlike']})
    info.append({'stolestotalrating':subMostrated.loc[17,'Count'],'stolesrated':subMostrated.loc[17,'Mean'],'stolestotalliking':subMostLiked.loc[17,'Countlike']})
    info.append({'tiestotalrating':subMostrated.loc[18,'Count'],'tiesrated':subMostrated.loc[18,'Mean'],'tiestotalliking':subMostLiked.loc[18,'Countlike']})
    info.append({'topweartotalrating':subMostrated.loc[19,'Count'],'topwearrated':subMostrated.loc[19,'Mean'],'topweartotalliking':subMostLiked.loc[19,'Countlike']})
    info.append({'childrentotalrating':subMostrated.loc[20,'Count'],'childrenrated':subMostrated.loc[20,'Mean'],'childrentotalliking':subMostLiked.loc[20,'Countlike']})
    info.append({'classicstotalrating':subMostrated.loc[21,'Count'],'classicsrated':subMostrated.loc[21,'Mean'],'classicstotalliking':subMostLiked.loc[21,'Countlike']})
    info.append({'crimetotalrating':subMostrated.loc[22,'Count'],'crimerated':subMostrated.loc[22,'Mean'],'crimetotalliking':subMostLiked.loc[22,'Countlike']})
    info.append({'families_relationshiptotalrating':subMostrated.loc[23,'Count'],'families_relationshiprated':subMostrated.loc[23,'Mean'],'families_relationshiptotalliking':subMostLiked.loc[23,'Countlike']})
    info.append({'fantasytotalrating':subMostrated.loc[24,'Count'],'fantasyrated':subMostrated.loc[24,'Mean'],'fantasytotalliking':subMostLiked.loc[24,'Countlike']})
    info.append({'generaltotalrating':subMostrated.loc[25,'Count'],'generalrated':subMostrated.loc[25,'Mean'],'generaltotalliking':subMostLiked.loc[25,'Countlike']})
    info.append({'literaturetotalrating':subMostrated.loc[26,'Count'],'literaturerated':subMostrated.loc[26,'Mean'],'literaturetotalliking':subMostLiked.loc[26,'Countlike']})
    info.append({'otherstotalrating':subMostrated.loc[27,'Count'],'othersrated':subMostrated.loc[27,'Mean'],'otherstotalliking':subMostLiked.loc[27,'Countlike']})
    info.append({'romancetotalrating':subMostrated.loc[28,'Count'],'romancerated':subMostrated.loc[28,'Mean'],'romancetotalliking':subMostLiked.loc[28,'Countlike']})
    info.append({'science_fictiontotalrating':subMostrated.loc[29,'Count'],'science_fictionrated':subMostrated.loc[29,'Mean'],'science_fictiontotalliking':subMostLiked.loc[29,'Countlike']})
    info.append({'short_storiestotalrating':subMostrated.loc[30,'Count'],'short_storiesrated':subMostrated.loc[30,'Mean'],'short_storiestotalliking':subMostLiked.loc[30,'Countlike']})
    info.append({'young_adultstotalrating':subMostrated.loc[31,'Count'],'young_adultsrated':subMostrated.loc[31,'Mean'],'young_adultstotalliking':subMostLiked.loc[31,'Countlike']})
    info.append({'flip_flopstotalrating':subMostrated.loc[32,'Count'],'flip_flopsrated':subMostrated.loc[32,'Mean'],'flip_flopstotalliking':subMostLiked.loc[32,'Countlike']})
    info.append({'sandaltotalrating':subMostrated.loc[33,'Count'],'sandalrated':subMostrated.loc[33,'Mean'],'sandaltotalliking':subMostLiked.loc[33,'Countlike']})
    info.append({'shoe_accessoriestotalrating':subMostrated.loc[34,'Count'],'shoe_accessoriesrated':subMostrated.loc[34,'Mean'],'shoe_accessoriestotalliking':subMostLiked.loc[34,'Countlike']})
    info.append({'shoestotalrating':subMostrated.loc[35,'Count'],'shoesrated':subMostrated.loc[35,'Mean'],'shoestotalliking':subMostLiked.loc[35,'Countlike']})
    info.append({'sockstotalrating':subMostrated.loc[36,'Count'],'socksrated':subMostrated.loc[36,'Mean'],'sockstotalliking':subMostLiked.loc[36,'Countlike']})
    info.append({'bath_bodytotalrating':subMostrated.loc[37,'Count'],'bath_bodyrated':subMostrated.loc[37,'Mean'],'bath_bodytotalliking':subMostLiked.loc[37,'Countlike']})
    info.append({'eyestotalrating':subMostrated.loc[38,'Count'],'eyesrated':subMostrated.loc[38,'Mean'],'eyestotalliking':subMostLiked.loc[38,'Countlike']})
    info.append({'fragrancetotalrating':subMostrated.loc[39,'Count'],'fragrancerated':subMostrated.loc[39,'Mean'],'fragrancetotalliking':subMostLiked.loc[39,'Countlike']})
    info.append({'hairtotalrating':subMostrated.loc[40,'Count'],'hairrated':subMostrated.loc[40,'Mean'],'hairtotalliking':subMostLiked.loc[40,'Countlike']})
    info.append({'lipstotalrating':subMostrated.loc[41,'Count'],'lipsrated':subMostrated.loc[41,'Mean'],'lipstotalliking':subMostLiked.loc[41,'Countlike']})
    info.append({'makeuptotalrating':subMostrated.loc[42,'Count'],'makeuprated':subMostrated.loc[42,'Mean'],'makeuptotalliking':subMostLiked.loc[42,'Countlike']})
    info.append({'nailstotalrating':subMostrated.loc[43,'Count'],'nailsrated':subMostrated.loc[43,'Mean'],'nailstotalliking':subMostLiked.loc[43,'Countlike']})
    info.append({'skin_caretotalrating':subMostrated.loc[44,'Count'],'skin_carerated':subMostrated.loc[44,'Mean'],'skin_caretotalliking':subMostLiked.loc[44,'Countlike']})
    info.append({'androidtotalrating':subMostrated.loc[45,'Count'],'androidrated':subMostrated.loc[45,'Mean'],'androidtotalliking':subMostLiked.loc[45,'Countlike']})
    info.append({'appletotalrating':subMostrated.loc[46,'Count'],'applerated':subMostrated.loc[46,'Mean'],'appletotalliking':subMostLiked.loc[46,'Countlike']})
    info.append({'honortotalrating':subMostrated.loc[47,'Count'],'honorrated':subMostrated.loc[47,'Mean'],'honortotalliking':subMostLiked.loc[47,'Countlike']})
    info.append({'huaweitotalrating':subMostrated.loc[48,'Count'],'huaweirated':subMostrated.loc[48,'Mean'],'huaweitotalliking':subMostLiked.loc[48,'Countlike']})
    info.append({'nokiatotalrating':subMostrated.loc[49,'Count'],'nokiarated':subMostrated.loc[49,'Mean'],'nokiatotalliking':subMostLiked.loc[49,'Countlike']})
    info.append({'oneplustotalrating':subMostrated.loc[50,'Count'],'oneplusrated':subMostrated.loc[50,'Mean'],'oneplustotalliking':subMostLiked.loc[50,'Countlike']})
    info.append({'oppototalrating':subMostrated.loc[51,'Count'],'opporated':subMostrated.loc[51,'Mean'],'oppototalliking':subMostLiked.loc[51,'Countlike']})
    info.append({'redmitotalrating':subMostrated.loc[52,'Count'],'redmirated':subMostrated.loc[52,'Mean'],'redmitotalliking':subMostLiked.loc[52,'Countlike']})
    info.append({'samsungtotalrating':subMostrated.loc[53,'Count'],'samsungrated':subMostrated.loc[53,'Mean'],'samsungtotalliking':subMostLiked.loc[53,'Countlike']})
    info.append({'sonytotalrating':subMostrated.loc[54,'Count'],'sonyrated':subMostrated.loc[54,'Mean'],'sonytotalliking':subMostLiked.loc[54,'Countlike']})
    info.append({'vivototalrating':subMostrated.loc[55,'Count'],'vivorated':subMostrated.loc[55,'Mean'],'vivototalliking':subMostLiked.loc[55,'Countlike']})
    info.append({'bags_Sportingtotalrating':subMostrated.loc[56,'Count'],'bags_Sportingrated':subMostrated.loc[56,'Mean'],'bags_Sportingtotalliking':subMostLiked.loc[56,'Countlike']})
    info.append({'eyewear_Sportingtotalrating':subMostrated.loc[57,'Count'],'eyewear_Sportingrated':subMostrated.loc[57,'Mean'],'eyewear_Sportingtotalliking':subMostLiked.loc[57,'Countlike']})
    info.append({'headwear_Sportingtotalrating':subMostrated.loc[58,'Count'],'headwear_Sportingrated':subMostrated.loc[58,'Mean'],'headwear_Sportingtotalliking':subMostLiked.loc[58,'Countlike']})
    info.append({'sports_equipmenttotalrating':subMostrated.loc[59,'Count'],'sports_equipmentrated':subMostrated.loc[59,'Mean'],'sports_equipmenttotalliking':subMostLiked.loc[59,'Countlike']})
    info.append({'watches_Sportingtotalrating':subMostrated.loc[60,'Count'],'watches_Sportingrated':subMostrated.loc[60,'Mean'],'watches_Sportingtotalliking':subMostLiked.loc[60,'Countlike']})
    info.append({'water_bottletotalrating':subMostrated.loc[61,'Count'],'water_bottlerated':subMostrated.loc[61,'Mean'],'water_bottletotalliking':subMostLiked.loc[61,'Countlike']})

    return jsonify(info)

@app.route('/api/info/<id>', methods=['GET'])
def informationsuser(id):
    tasks = mongo.db.products
    id =  int(id)

    info = []
    #count Categories
    Accessories = tasks.count_documents( {"Categories" : "Accessories","vendeurId" : id} )
    Apparel = tasks.find( {"Categories" : "Apparel","vendeurId" : id} ).count()
    Books = tasks.find( {"Categories" : "Books","vendeurId" : id} ).count()
    Footwear = tasks.find( {"Categories" : "Footwear","vendeurId" : id} ).count()
    Personal_Care = tasks.find( {"Categories" : "Personal Care","vendeurId" : id} ).count()
    Phones = tasks.find( {"Categories" : "Phones","vendeurId" : id} ).count()
    Sporting = tasks.find( {"Categories" : "Sporting Goods","vendeurId" : id} ).count()
    #count subcategories
    belts = tasks.find( {"subCategory" : "belts","vendeurId" : id} ).count()
    jewellery = tasks.find( {"subCategory" : "jewellery","vendeurId" : id} ).count()
    eyewear_Accessories = tasks.find( {"Categories" : "Accessories","subCategory" : "eyewear","vendeurId" : id} ).count()
    bags_Accessories = tasks.find( {"Categories" : "Accessories","subCategory" : "bags","vendeurId" : id} ).count()
    headwear_Accessories = tasks.find( {"Categories" : "Accessories","subCategory" : "headwear","vendeurId" : id} ).count()
    wallets = tasks.find( {"subCategory" : "wallets","vendeurId" : id} ).count()
    cufflinks = tasks.find( {"subCategory" : "cufflinks","vendeurId" : id} ).count()
    gloves = tasks.find( {"subCategory" : "gloves","vendeurId" : id} ).count()
    watches_Accessories = tasks.find( {"Categories" : "Accessories","subCategory" : "watches","vendeurId" : id} ).count()
    gift = tasks.find( {"subCategory" : "accessory gift set","vendeurId" : id} ).count()
    innerwear = tasks.find( {"subCategory" : "innerwear","VendeurId" : id} ).count()
    scarves = tasks.find( {"subCategory" : "scarves","VendeurId" : id} ).count()
    bottomwear = tasks.find( {"subCategory" : "bottomwear","vendeurId" : id} ).count()
    topwear = tasks.find( {"subCategory" : "topwear","vendeurId" : id} ).count()
    apparel_set = tasks.find( {"subCategory" : "apparel set","vendeurId" : id} ).count()
    dress = tasks.find( {"subCategory" : "dress","vendeurId" : id} ).count()
    ties = tasks.find( {"subCategory" : "ties","vendeurId" : id} ).count()
    stoles = tasks.find( {"subCategory" : "stoles","vendeurId" : id} ).count()
    loungewear_nightwear = tasks.find( {"subCategory" : "loungewear and nightwear","vendeurId" : id} ).count()
    saree = tasks.find( {"subCategory" : "saree","vendeurId" : id} ).count()
    general = tasks.find( {"subCategory" : "general","vendeurId" : id} ).count()
    others = tasks.find( {"subCategory" : "others","vendeurId" : id} ).count()
    literature = tasks.find( {"subCategory" : "literature","vendeurId" : id} ).count()
    families_relationship = tasks.find( {"subCategory" : "families & relationship","vendeurId" : id} ).count()
    fantasy = tasks.find( {"subCategory" : "fantasy","vendeurId" : id} ).count()
    children = tasks.find( {"subCategory" : "children","vendeurId" : id} ).count()
    classics = tasks.find( {"subCategory" : "classics","vendeurId" : id} ).count()
    crime = tasks.find( {"subCategory" : "crime","vendeurId" : id} ).count()
    short_stories = tasks.find( {"subCategory" : "short stories","vendeurId" : id} ).count()
    young_adults = tasks.find( {"subCategory" : "young adults","vendeurId" : id} ).count()
    romance = tasks.find( {"subCategory" : "romance","vendeurId" : id} ).count()
    science_fiction = tasks.find( {"subCategory" : "science fiction","vendeurId" : id} ).count()
    socks = tasks.find( {"subCategory" : "socks","vendeurId" : id} ).count()
    shoes = tasks.find( {"subCategory" : "shoes","VendeurId" : id} ).count()
    flip_flops = tasks.find( {"subCategory" : "flip flops","VendeurId" : id} ).count()
    sandal = tasks.find( {"subCategory" : "sandal","vendeurId" : id} ).count()
    shoe_accessories = tasks.find( {"subCategory" : "shoe accessories","vendeurId" : id} ).count()
    fragrance = tasks.find( {"subCategory" : "fragrance","VendeurId" : id} ).count()
    lips = tasks.find( {"subCategory" : "lips","vendeurId" : id} ).count()
    eyes = tasks.find( {"subCategory" : "eyes","vendeurId" : id} ).count()
    hair = tasks.find( {"subCategory" : "hair","vendeurId" : id} ).count()
    makeup = tasks.find( {"subCategory" : "makeup","vendeurId" : id} ).count()
    bath_body = tasks.find( {"subCategory" : "bath and body","vendeurId" : id} ).count()
    skin_care = tasks.find( {"subCategory" : "skin care","vendeurId" : id} ).count()
    nails = tasks.find( {"subCategory" : "nails","vendeurId" : id} ).count()
    android = tasks.find( {"subCategory" : "android","vendeurId" : id} ).count()
    honor = tasks.find( {"subCategory" : "honor","vendeurId" : id} ).count()
    vivo = tasks.find( {"subCategory" : "vivo","vendeurId" : id} ).count()
    sony = tasks.find( {"subCategory" : "sony","vendeurId" : id} ).count()
    oppo = tasks.find( {"subCategory" : "oppo","vendeurId" : id} ).count()
    redmi = tasks.find( {"subCategory" : "redmi","vendeurId" : id} ).count()
    nokia = tasks.find( {"subCategory" : "nokia","vendeurId" : id} ).count()
    oneplus = tasks.find( {"subCategory" : "oneplus","vendeurId" : id} ).count()
    apple = tasks.find( {"subCategory" : "apple","vendeurId" : id} ).count()
    samsung = tasks.find( {"subCategory" : "samsung","vendeurId" : id} ).count()
    huawei = tasks.find( {"subCategory" : "huawei","vendeurId" : id} ).count()
    bags_Sporting = tasks.find( {"Categories" : "Sporting Goods","subCategory" : "bags","vendeurId" : id} ).count()
    headwear_Sporting = tasks.find( {"Categories" : "Sporting Goods","subCategory" : "headwear","vendeurId" : id} ).count()
    water_bottle = tasks.find( {"subCategory" : "water bottle","vendeurId" : id} ).count()
    eyewear_Sporting = tasks.find( {"Categories" : "Sporting Goods","subCategory" : "eyewear","vendeurId" : id} ).count()
    watches_Sporting = tasks.find( {"Categories" : "Sporting Goods","subCategory" : "watches","vendeurId" : id} ).count()
    sports_equipment = tasks.find( {"subCategory" : "sports equipment","vendeurId" : id} ).count()
     
    info.append({'accessoriescount':Accessories,'apparelcount':Apparel,'bookscount':Books,'footwearcount':Footwear,'personal_Carecount':Personal_Care,'phonescount':Phones,'sportcount':Sporting})  
    info.append({'beltscount':belts,'jewellerycount':jewellery,'eyewear_Accessoriescount':eyewear_Accessories,'bags_Accessoriescount':bags_Accessories,'headwear_Accessoriescount':headwear_Accessories,'walletscount':wallets,'cufflinkscount':cufflinks,'glovescount':gloves,'watches_Accessoriescount':watches_Accessories,'giftcount':gift})
    info.append({'innerwearcount':innerwear,'scarvescount':scarves,'bottomwearcount':bottomwear,'topwearcount':topwear,'apparel_setcount':apparel_set,'dresscount':dress,'tiescount':ties,'stolescount':stoles,'loungewear_nightwearcount':loungewear_nightwear,'sareecount':saree})
    info.append({'generalcount':general,'otherscount':others,'literaturecount':literature,'families_relationshipcount':families_relationship,'fantasycount':fantasy,'childrencount':children,'classicscount':classics,'crimecount':crime,'short_storiescount':short_stories,'young_adultscount':young_adults,'romancecount':romance,'science_fictioncount':science_fiction})
    info.append({'sockscount':socks,'shoescount':shoes,'flip_flopscount':flip_flops,'sandalcount':sandal,'shoe_accessoriescount':shoe_accessories})
    info.append({'fragrancecount':fragrance,'lipscount':lips,'eyecount':eyes,'haircount':hair,'makeupcount':makeup,'bath_bodycount':bath_body,'skin_carecount':skin_care,'nailscount':nails})
    info.append({'androidcount':android,'honorcount':honor,'vivocount':vivo,'sonycount':sony,'oppocount':oppo,'redmicount':redmi,'nokiacount':nokia,'onepluscount':oneplus,'applecount':apple,'samsungcount':samsung,'huaweicount':huawei})
    info.append({'bags_Sportingcount':bags_Sporting,'headwear_Sportingcount':headwear_Sporting,'water_bottlecount':water_bottle,'eyewear_Sportingcount':eyewear_Sporting,'watches_Sportingcount':watches_Sporting,'sports_equipmentcount':sports_equipment})
    
    #Most rated by category
    id =  str(id)
    dfratinguser = dfrating.loc[dfrating['vendeurId'] == id]
    countrating = dfratinguser.groupby("Categories", as_index=False).count()
    
    meanrating = dfratinguser.groupby("Categories", as_index=False).mean()
    Mostrated = pd.merge(countrating, meanrating, how='right', on=['Categories'])
    Mostrated["Count"] = Mostrated["rating_x"]
    Mostrated["Mean"] = Mostrated["rating_y"]
    Mostrated = Mostrated[['Categories',"Count","Mean"]]
    Mostrated['Count'] = Mostrated['Count'].astype(str)
    Mostrated['Mean'] = Mostrated['Mean'].astype(str)
    
    #Most liked by category
    id =  str(id)
    dflikeuser = dflike.loc[dflike['vendeurId'] == id]
    countlike = dflikeuser.groupby("Categories", as_index=False).count()
    countlike["Countlike"] = countlike["like"]
    MostLiked = countlike[['Categories',"Countlike"]]
    MostLiked['Countlike'] = MostLiked['Countlike'].astype(str)
    
    if ((dfratinguser.Categories == 'Accessories').any()) == True and ((dflikeuser.Categories == 'Accessories').any()) == True:
        info.append({'accessoriestotalrating':Mostrated.loc[Mostrated['Categories'] == 'Accessories','Count'].values[0],'accessoriesrated':Mostrated.loc[Mostrated['Categories'] == 'Accessories','Mean'].values[0],'accessoriestotalliking':MostLiked.loc[Mostrated['Categories'] == 'Accessories','Countlike'].values[0]})
    else:
        info.append({'accessoriestotalrating':0,'accessoriesrated':0,'accessoriestotalliking':0})
    if ((dfratinguser.Categories == 'Apparel').any()) == True and ((dflikeuser.Categories == 'Apparel').any()) == True:
        info.append({'appareltotalrating':Mostrated.loc[Mostrated['Categories'] == 'Apparel','Count'].values[0],'apparelrated':Mostrated.loc[Mostrated['Categories'] == 'Apparel','Mean'].values[0],'appareltotalliking':MostLiked.loc[Mostrated['Categories'] == 'Apparel','Countlike'].values[0]})
    else:
        info.append({'appareltotalrating':0,'apparelrated':0,'appareltotalliking':0})
    if ((dfratinguser.Categories == 'Books').any()) == True and ((dflikeuser.Categories == 'Books').any()) == True:
        info.append({'bookstotalrating':Mostrated.loc[Mostrated['Categories'] == 'Books','Count'].values[0],'booksrated':Mostrated.loc[Mostrated['Categories'] == 'Books','Mean'].values[0],'bookstotalliking':MostLiked.loc[Mostrated['Categories'] == 'Books','Countlike'].values[0]})
    else:
        info.append({'bookstotalrating':0,'booksrated':0,'bookstotalliking':0})
    if ((dfratinguser.Categories == 'Footwear').any()) == True and ((dflikeuser.Categories == 'Footwear').any()) == True:
        info.append({'footweartotalrating':Mostrated.loc[Mostrated['Categories'] == 'Footwear','Count'].values[0],'footwearrated':Mostrated.loc[Mostrated['Categories'] == 'Footwear','Mean'].values[0],'footweartotalliking':MostLiked.loc[Mostrated['Categories'] == 'Footwear','Countlike'].values[0]})
    else:
        info.append({'footweartotalrating':0,'footwearrated':0,'footweartotalliking':0})
    if ((dfratinguser.Categories == 'Personal Care').any()) == True and ((dflikeuser.Categories == 'Personal Care').any()) == True:
        info.append({'personal_Caretotalrating':Mostrated.loc[Mostrated['Categories'] == 'Personal Care','Count'].values[0],'personal_Carerated':Mostrated.loc[Mostrated['Categories'] == 'Personal Care','Mean'].values[0],'personal_Caretotalliking':MostLiked.loc[Mostrated['Categories'] == 'Personal Care','Countlike'].values[0]})
    else:
        info.append({'personal_Caretotalrating':0,'personal_Carerated':0,'personal_Caretotalliking':0})
    if ((dfratinguser.Categories == 'Phones').any()) == True and ((dflikeuser.Categories == 'Phones').any()) == True:
        info.append({'phonestotalrating':Mostrated.loc[Mostrated['Categories'] == 'Phones','Count'].values[0],'phonesrated':Mostrated.loc[Mostrated['Categories'] == 'Phones','Mean'].values[0],'phonestotalliking':MostLiked.loc[Mostrated['Categories'] == 'Phones','Countlike'].values[0]})
    else:
        info.append({'phonestotalrating':0,'phonesrated':0,'phonestotalliking':0})
    if ((dfratinguser.Categories == 'Sporting Goods').any()) == True and ((dflikeuser.Categories == 'Sporting Goods').any()) == True:
        info.append({'sporttotalrating':Mostrated.loc[Mostrated['Categories'] == 'Sporting Goods','Count'].values[0],'sportrated':Mostrated.loc[Mostrated['Categories'] == 'Sporting Goods','Mean'].values[0],'sporttotalliking':MostLiked.loc[Mostrated['Categories'] == 'Sporting Goods','Countlike'].values[0]})
    else:
        info.append({'sporttotalrating':0,'sportrated':0,'sporttotalliking':0})
      
    #Most rated by subcategory
    id =  str(id)
    dfratinguser = dfrating.loc[dfrating['vendeurId'] == id]
    subcountrating = dfratinguser.groupby(['Categories','subCategory'], as_index=False).count()
    submeanrating = dfratinguser.groupby(['Categories','subCategory'], as_index=False).mean()
    subMostrated = pd.merge(subcountrating, submeanrating, how='left', on=['Categories','subCategory'])
    subMostrated["Count"] = subMostrated["rating_x"]
    subMostrated["Mean"] = subMostrated["rating_y"]
    subMostrated = subMostrated[['Categories','subCategory',"Count","Mean"]]
    subMostrated['Count'] = subMostrated['Count'].astype(str)
    subMostrated['Mean'] = subMostrated['Mean'].astype(str)
    
    #Most liked by subcategory
    id =  str(id)
    dflikeuser = dflike.loc[dflike['vendeurId'] == id]
    subcountlike = dflikeuser.groupby(['Categories','subCategory'], as_index=False).count()
    subcountlike["Countlike"] = subcountlike["like"]
    subMostLiked = subcountlike[['Categories','subCategory',"Countlike"]]
    subMostLiked['Countlike'] = subMostLiked['Countlike'].astype(str)
    
    if (((dfratinguser.Categories == 'Accessories') & (dfratinguser.subCategory == 'accessory gift set')).any()) == True and (((dflikeuser.Categories == 'Accessories') & (dflikeuser.subCategory == 'accessory gift set')).any()) == True:
        info.append({'gifttotalrating':subMostrated.loc[(subMostrated['Categories'] == 'Accessories') & (subMostrated['subCategory'] == 'accessory gift set'),'Count'].values[0],'giftsrated':subMostrated.loc[(subMostrated['Categories'] == 'Accessories') & (subMostrated['subCategory'] == 'accessory gift set'),'Mean'].values[0],'giftstotalliking':subMostLiked.loc[(subMostrated['Categories'] == 'Accessories') & (subMostrated['subCategory'] == 'accessory gift set'),'Countlike'].values[0]})
    else:
        info.append({'gifttotalrating':0,'giftsrated':0,'giftstotalliking':0})
    if (((dfratinguser.Categories == 'Accessories') & (dfratinguser.subCategory == 'bags')).any()) == True and (((dflikeuser.Categories == 'Accessories') & (dflikeuser.subCategory == 'bags')).any()) == True:
        info.append({'bags_Accessoriestotalrating':subMostrated.loc[(subMostrated['Categories'] == 'Accessories') & (subMostrated['subCategory'] == 'bags'),'Count'].values[0],'bags_Accessoriesrated':subMostrated.loc[(subMostrated['Categories'] == 'Accessories') & (subMostrated['subCategory'] == 'bags'),'Mean'].values[0],'bags_Accessoriestotalliking':subMostLiked.loc[(subMostrated['Categories'] == 'Accessories') & (subMostrated['subCategory'] == 'bags'),'Countlike'].values[0]})
    else:
        info.append({'bags_Accessoriestotalrating':0,'bags_Accessoriesrated':0,'bags_Accessoriestotalliking':0})
    if (((dfratinguser.Categories == 'Accessories') & (dfratinguser.subCategory == 'belts')).any()) == True and (((dflikeuser.Categories == 'Accessories') & (dflikeuser.subCategory == 'belts')).any()) == True:
        info.append({'beltstotalrating':subMostrated.loc[(subMostrated['Categories'] == 'Accessories') & (subMostrated['subCategory'] == 'belts'),'Count'].values[0],'beltsrated':subMostrated.loc[(subMostrated['Categories'] == 'Accessories') & (subMostrated['subCategory'] == 'belts'),'Mean'].values[0],'beltstotalliking':subMostLiked.loc[(subMostrated['Categories'] == 'Accessories') & (subMostrated['subCategory'] == 'belts'),'Countlike'].values[0]})
    else:
        info.append({'beltstotalrating':0,'beltsrated':0,'beltstotalliking':0})
    if (((dfratinguser.Categories == 'Accessories') & (dfratinguser.subCategory == 'cufflinks')).any()) == True and (((dflikeuser.Categories == 'Accessories') & (dflikeuser.subCategory == 'cufflinks')).any()) == True:
        info.append({'cufflinkstotalrating':subMostrated.loc[(subMostrated['Categories'] == 'Accessories') & (subMostrated['subCategory'] == 'cufflinks'),'Count'].values[0],'cufflinksrated':subMostrated.loc[(subMostrated['Categories'] == 'Accessories') & (subMostrated['subCategory'] == 'cufflinks'),'Mean'].values[0],'cufflinkstotalliking':subMostLiked.loc[(subMostrated['Categories'] == 'Accessories') & (subMostrated['subCategory'] == 'cufflinks'),'Countlike'].values[0]})
    else:
        info.append({'cufflinkstotalrating':0,'cufflinksrated':0,'cufflinkstotalliking':0})
    if (((dfratinguser.Categories == 'Accessories') & (dfratinguser.subCategory == 'eyewear')).any()) == True and (((dflikeuser.Categories == 'Accessories') & (dflikeuser.subCategory == 'eyewear')).any()) == True:
        info.append({'eyewear_Accessoriestotalrating':subMostrated.loc[(subMostrated['Categories'] == 'Accessories') & (subMostrated['subCategory'] == 'eyewear'),'Count'].values[0],'eyewear_Accessoriesrated':subMostrated.loc[(subMostrated['Categories'] == 'Accessories') & (subMostrated['subCategory'] == 'eyewear'),'Mean'].values[0],'eyewear_Accessoriestotalliking':subMostLiked.loc[(subMostrated['Categories'] == 'Accessories') & (subMostrated['subCategory'] == 'eyewear'),'Countlike'].values[0]})
    else:
        info.append({'eyewear_Accessoriestotalrating':0,'eyewear_Accessoriesrated':0,'eyewear_Accessoriestotalliking':0}) 
    if (((dfratinguser.Categories == 'Accessories') & (dfratinguser.subCategory == 'gloves')).any()) == True and (((dflikeuser.Categories == 'Accessories') & (dflikeuser.subCategory == 'gloves')).any()) == True:
        info.append({'glovestotalrating':subMostrated.loc[(subMostrated['Categories'] == 'Accessories') & (subMostrated['subCategory'] == 'gloves'),'Count'].values[0],'glovesrated':subMostrated.loc[(subMostrated['Categories'] == 'Accessories') & (subMostrated['subCategory'] == 'gloves'),'Mean'].values[0],'glovestotalliking':subMostLiked.loc[(subMostrated['Categories'] == 'Accessories') & (subMostrated['subCategory'] == 'gloves'),'Countlike'].values[0]})
    else:
        info.append({'glovestotalrating':0,'glovesrated':0,'glovestotalliking':0})
    if (((dfratinguser.Categories == 'Accessories') & (dfratinguser.subCategory == 'headwear')).any()) == True and (((dflikeuser.Categories == 'Accessories') & (dflikeuser.subCategory == 'headwear')).any()) == True:
        info.append({'headwear_Accessoriestotalrating':subMostrated.loc[(subMostrated['Categories'] == 'Accessories') & (subMostrated['subCategory'] == 'headwear'),'Count'].values[0],'headwear_Accessoriesrated':subMostrated.loc[(subMostrated['Categories'] == 'Accessories') & (subMostrated['subCategory'] == 'headwear'),'Mean'].values[0],'headwear_Accessoriestotalliking':subMostLiked.loc[(subMostrated['Categories'] == 'Accessories') & (subMostrated['subCategory'] == 'headwear'),'Countlike'].values[0]})
    else:
        info.append({'headwear_Accessoriestotalrating':0,'headwear_Accessoriesrated':0,'headwear_Accessoriestotalliking':0})
    if (((dfratinguser.Categories == 'Accessories') & (dfratinguser.subCategory == 'jewellery')).any()) == True and (((dflikeuser.Categories == 'Accessories') & (dflikeuser.subCategory == 'jewellery')).any()) == True:
        info.append({'jewellerytotalrating':subMostrated.loc[(subMostrated['Categories'] == 'Accessories') & (subMostrated['subCategory'] == 'jewellery'),'Count'].values[0],'jewelleryrated':subMostrated.loc[(subMostrated['Categories'] == 'Accessories') & (subMostrated['subCategory'] == 'jewellery'),'Mean'].values[0],'jewellerytotalliking':subMostLiked.loc[(subMostrated['Categories'] == 'Accessories') & (subMostrated['subCategory'] == 'jewellery'),'Countlike'].values[0]})
    else:
        info.append({'jewellerytotalrating':0,'jewelleryrated':0,'jewellerytotalliking':0})
    if (((dfratinguser.Categories == 'Accessories') & (dfratinguser.subCategory == 'wallets')).any()) == True and (((dflikeuser.Categories == 'Accessories') & (dflikeuser.subCategory == 'wallets')).any()) == True:
        info.append({'walletstotalrating':subMostrated.loc[(subMostrated['Categories'] == 'Accessories') & (subMostrated['subCategory'] == 'wallets'),'Count'].values[0],'walletsrated':subMostrated.loc[(subMostrated['Categories'] == 'Accessories') & (subMostrated['subCategory'] == 'wallets'),'Mean'].values[0],'walletstotalliking':subMostLiked.loc[(subMostrated['Categories'] == 'Accessories') & (subMostrated['subCategory'] == 'wallets'),'Countlike'].values[0]})
    else:
        info.append({'walletstotalrating':0,'walletsrated':0,'walletstotalliking':0})
    if (((dfratinguser.Categories == 'Accessories') & (dfratinguser.subCategory == 'watches')).any()) == True and (((dflikeuser.Categories == 'Accessories') & (dflikeuser.subCategory == 'watches')).any()) == True:
        info.append({'watches_Accessoriestotalrating':subMostrated.loc[(subMostrated['Categories'] == 'Accessories') & (subMostrated['subCategory'] == 'watches'),'Count'].values[0],'watches_Accessoriesrated':subMostrated.loc[(subMostrated['Categories'] == 'Accessories') & (subMostrated['subCategory'] == 'watches'),'Mean'].values[0],'watches_Accessoriestotalliking':subMostLiked.loc[(subMostrated['Categories'] == 'Accessories') & (subMostrated['subCategory'] == 'watches'),'Countlike'].values[0]})
    else:
        info.append({'watches_Accessoriestotalrating':0,'watches_Accessoriesrated':0,'watches_Accessoriestotalliking':0})
    if (((dfratinguser.Categories == 'Apparel') & (dfratinguser.subCategory == 'apparel set')).any()) == True and (((dflikeuser.Categories == 'Apparel') & (dflikeuser.subCategory == 'apparel set')).any()) == True:
        info.append({'apparel_settotalrating':subMostrated.loc[(subMostrated['Categories'] == 'Apparel') & (subMostrated['subCategory'] == 'apparel set'),'Count'].values[0],'apparel_setrated':subMostrated.loc[(subMostrated['Categories'] == 'Apparel') & (subMostrated['subCategory'] == 'apparel set'),'Mean'].values[0],'apparel_settotalliking':subMostLiked.loc[(subMostrated['Categories'] == 'Apparel') & (subMostrated['subCategory'] == 'apparel set'),'Countlike'].values[0]})
    else:
        info.append({'apparel_settotalrating':0,'apparel_setrated':0,'apparel_settotalliking':0})
    if (((dfratinguser.Categories == 'Apparel') & (dfratinguser.subCategory == 'bottomwear')).any()) == True and (((dflikeuser.Categories == 'Apparel') & (dflikeuser.subCategory == 'bottomwear')).any()) == True:
        info.append({'bottomweatotalrrating':subMostrated.loc[(subMostrated['Categories'] == 'Apparel') & (subMostrated['subCategory'] == 'bottomwear'),'Count'].values[0],'bottomwearrated':subMostrated.loc[(subMostrated['Categories'] == 'Apparel') & (subMostrated['subCategory'] == 'bottomwear'),'Mean'].values[0],'bottomweartotalliking':subMostLiked.loc[(subMostrated['Categories'] == 'Apparel') & (subMostrated['subCategory'] == 'bottomwear'),'Countlike'].values[0]})
    else:
        info.append({'bottomweatotalrrating':0,'bottomwearrated':0,'bottomweartotalliking':0})
    if (((dfratinguser.Categories == 'Apparel') & (dfratinguser.subCategory == 'dress')).any()) == True and (((dflikeuser.Categories == 'Apparel') & (dflikeuser.subCategory == 'dress')).any()) == True:
        info.append({'dresstotalrating':subMostrated.loc[(subMostrated['Categories'] == 'Apparel') & (subMostrated['subCategory'] == 'dress'),'Count'].values[0],'dressrated':subMostrated.loc[(subMostrated['Categories'] == 'Apparel') & (subMostrated['subCategory'] == 'dress'),'Mean'].values[0],'dresstotalliking':subMostLiked.loc[(subMostrated['Categories'] == 'Apparel') & (subMostrated['subCategory'] == 'dress'),'Countlike'].values[0]})
    else:
        info.append({'dresstotalrating':0,'dressrated':0,'dresstotalliking':0}) 
    if (((dfratinguser.Categories == 'Apparel') & (dfratinguser.subCategory == 'innerwear')).any()) == True and (((dflikeuser.Categories == 'Apparel') & (dflikeuser.subCategory == 'innerwear')).any()) == True:
        info.append({'innerweartotalrating':subMostrated.loc[(subMostrated['Categories'] == 'Apparel') & (subMostrated['subCategory'] == 'innerwear'),'Count'].values[0],'innerwearrated':subMostrated.loc[(subMostrated['Categories'] == 'Apparel') & (subMostrated['subCategory'] == 'innerwear'),'Mean'].values[0],'innerweartotalliking':subMostLiked.loc[(subMostrated['Categories'] == 'Apparel') & (subMostrated['subCategory'] == 'innerwear'),'Countlike'].values[0]})
    else:
        info.append({'innerweartotalrating':0,'innerwearrated':0,'innerweartotalliking':0})
    if (((dfratinguser.Categories == 'Apparel') & (dfratinguser.subCategory == 'loungewear and nightwear')).any()) == True and (((dflikeuser.Categories == 'Apparel') & (dflikeuser.subCategory == 'loungewear and nightwear')).any()) == True:
        info.append({'loungewear_nightweartotalrating':subMostrated.loc[(subMostrated['Categories'] == 'Apparel') & (subMostrated['subCategory'] == 'loungewear and nightwear'),'Count'].values[0],'loungewear_nightwearrated':subMostrated.loc[(subMostrated['Categories'] == 'Apparel') & (subMostrated['subCategory'] == 'loungewear and nightwear'),'Mean'].values[0],'loungewear_nightweartotalliking':subMostLiked.loc[(subMostrated['Categories'] == 'Apparel') & (subMostrated['subCategory'] == 'loungewear and nightwear'),'Countlike'].values[0]})
    else:
        info.append({'loungewear_nightweartotalrating':0,'loungewear_nightwearrated':0,'loungewear_nightweartotalliking':0})
    if (((dfratinguser.Categories == 'Apparel') & (dfratinguser.subCategory == 'saree')).any()) == True and (((dflikeuser.Categories == 'Apparel') & (dflikeuser.subCategory == 'saree')).any()) == True:
        info.append({'sareetotalrating':subMostrated.loc[(subMostrated['Categories'] == 'Apparel') & (subMostrated['subCategory'] == 'saree'),'Count'].values[0],'sareerated':subMostrated.loc[(subMostrated['Categories'] == 'Apparel') & (subMostrated['subCategory'] == 'saree'),'Mean'].values[0],'sareetotalliking':subMostLiked.loc[(subMostrated['Categories'] == 'Apparel') & (subMostrated['subCategory'] == 'saree'),'Countlike'].values[0]})
    else:
        info.append({'sareetotalrating':0,'sareerated':0,'sareetotalliking':0})
    if (((dfratinguser.Categories == 'Apparel') & (dfratinguser.subCategory == 'scarves')).any()) == True and (((dflikeuser.Categories == 'Apparel') & (dflikeuser.subCategory == 'scarves')).any()) == True:
        info.append({'scarvestotalrating':subMostrated.loc[(subMostrated['Categories'] == 'Apparel') & (subMostrated['subCategory'] == 'scarves'),'Count'].values[0],'scarvesrated':subMostrated.loc[(subMostrated['Categories'] == 'Apparel') & (subMostrated['subCategory'] == 'scarves'),'Mean'].values[0],'scarvestotalliking':subMostLiked.loc[(subMostrated['Categories'] == 'Apparel') & (subMostrated['subCategory'] == 'scarves'),'Countlike'].values[0]})
    else:
        info.append({'scarvestotalrating':0,'scarvesrated':0,'scarvestotalliking':0})
    if (((dfratinguser.Categories == 'Apparel') & (dfratinguser.subCategory == 'stoles')).any()) == True and (((dflikeuser.Categories == 'Apparel') & (dflikeuser.subCategory == 'stoles')).any()) == True:
        info.append({'stolestotalrating':subMostrated.loc[(subMostrated['Categories'] == 'Apparel') & (subMostrated['subCategory'] == 'stoles'),'Count'].values[0],'stolesrated':subMostrated.loc[(subMostrated['Categories'] == 'Apparel') & (subMostrated['subCategory'] == 'stoles'),'Mean'].values[0],'stolestotalliking':subMostLiked.loc[(subMostrated['Categories'] == 'Apparel') & (subMostrated['subCategory'] == 'stoles'),'Countlike'].values[0]})
    else:
        info.append({'stolestotalrating':0,'stolesrated':0,'stolestotalliking':0})
    if (((dfratinguser.Categories == 'Apparel') & (dfratinguser.subCategory == 'ties')).any()) == True and (((dflikeuser.Categories == 'Apparel') & (dflikeuser.subCategory == 'ties')).any()) == True:
        info.append({'tiestotalrating':subMostrated.loc[(subMostrated['Categories'] == 'Apparel') & (subMostrated['subCategory'] == 'ties'),'Count'].values[0],'tiesrated':subMostrated.loc[(subMostrated['Categories'] == 'Apparel') & (subMostrated['subCategory'] == 'ties'),'Mean'].values[0],'tiestotalliking':subMostLiked.loc[(subMostrated['Categories'] == 'Apparel') & (subMostrated['subCategory'] == 'ties'),'Countlike'].values[0]})
    else:
        info.append({'tiestotalrating':0,'tiesrated':0,'tiestotalliking':0})
    if (((dfratinguser.Categories == 'Apparel') & (dfratinguser.subCategory == 'topwear')).any()) == True and (((dflikeuser.Categories == 'Apparel') & (dflikeuser.subCategory == 'topwear')).any()) == True:
        info.append({'topweartotalrating':subMostrated.loc[(subMostrated['Categories'] == 'Apparel') & (subMostrated['subCategory'] == 'topwear'),'Count'].values[0],'topwearrated':subMostrated.loc[(subMostrated['Categories'] == 'Apparel') & (subMostrated['subCategory'] == 'topwear'),'Mean'].values[0],'topweartotalliking':subMostLiked.loc[(subMostrated['Categories'] == 'Apparel') & (subMostrated['subCategory'] == 'topwear'),'Countlike'].values[0]})
    else:
        info.append({'topweartotalrating':0,'topwearrated':0,'topweartotalliking':0}) 
    if (((dfratinguser.Categories == 'Books') & (dfratinguser.subCategory == 'children')).any()) == True and (((dflikeuser.Categories == 'Books') & (dflikeuser.subCategory == 'children')).any()) == True:
        info.append({'childrentotalrating':subMostrated.loc[(subMostrated['Categories'] == 'Books') & (subMostrated['subCategory'] == 'children'),'Count'].values[0],'childrenrated':subMostrated.loc[(subMostrated['Categories'] == 'Books') & (subMostrated['subCategory'] == 'children'),'Mean'].values[0],'childrentotalliking':subMostLiked.loc[(subMostrated['Categories'] == 'Books') & (subMostrated['subCategory'] == 'children'),'Countlike'].values[0]})
    else:
        info.append({'childrentotalrating':0,'childrenrated':0,'childrentotalliking':0})
    if (((dfratinguser.Categories == 'Books') & (dfratinguser.subCategory == 'classics')).any()) == True and (((dflikeuser.Categories == 'Books') & (dflikeuser.subCategory == 'classics')).any()) == True:
        info.append({'classicstotalrating':subMostrated.loc[(subMostrated['Categories'] == 'Books') & (subMostrated['subCategory'] == 'classics'),'Count'].values[0],'classicsrated':subMostrated.loc[(subMostrated['Categories'] == 'Books') & (subMostrated['subCategory'] == 'classics'),'Mean'].values[0],'classicstotalliking':subMostLiked.loc[(subMostrated['Categories'] == 'Books') & (subMostrated['subCategory'] == 'classics'),'Countlike'].values[0]})
    else:
        info.append({'classicstotalrating':0,'classicsrated':0,'classicstotalliking':0})
    if (((dfratinguser.Categories == 'Books') & (dfratinguser.subCategory == 'crime')).any()) == True and (((dflikeuser.Categories == 'Books') & (dflikeuser.subCategory == 'crime')).any()) == True:
        info.append({'crimetotalrating':subMostrated.loc[(subMostrated['Categories'] == 'Books') & (subMostrated['subCategory'] == 'crime'),'Count'].values[0],'crimerated':subMostrated.loc[(subMostrated['Categories'] == 'Books') & (subMostrated['subCategory'] == 'crime'),'Mean'].values[0],'crimetotalliking':subMostLiked.loc[(subMostrated['Categories'] == 'Books') & (subMostrated['subCategory'] == 'crime'),'Countlike'].values[0]})
    else:
        info.append({'crimetotalrating':0,'crimerated':0,'crimetotalliking':0})
    if (((dfratinguser.Categories == 'Books') & (dfratinguser.subCategory == 'families & relationship')).any()) == True and (((dflikeuser.Categories == 'Books') & (dflikeuser.subCategory == 'families & relationship')).any()) == True:
        info.append({'families_relationshiptotalrating':subMostrated.loc[(subMostrated['Categories'] == 'Books') & (subMostrated['subCategory'] == 'families & relationship'),'Count'].values[0],'families_relationshiprated':subMostrated.loc[(subMostrated['Categories'] == 'Books') & (subMostrated['subCategory'] == 'families & relationship'),'Mean'].values[0],'families_relationshiptotalliking':subMostLiked.loc[(subMostrated['Categories'] == 'Books') & (subMostrated['subCategory'] == 'families & relationship'),'Countlike'].values[0]})
    else:
        info.append({'families_relationshiptotalrating':0,'families_relationshiprated':0,'families_relationshiptotalliking':0})
    if (((dfratinguser.Categories == 'Books') & (dfratinguser.subCategory == 'fantasy')).any()) == True and (((dflikeuser.Categories == 'Books') & (dflikeuser.subCategory == 'fantasy')).any()) == True:
        info.append({'fantasytotalrating':subMostrated.loc[(subMostrated['Categories'] == 'Books') & (subMostrated['subCategory'] == 'fantasy'),'Count'].values[0],'fantasyrated':subMostrated.loc[(subMostrated['Categories'] == 'Books') & (subMostrated['subCategory'] == 'fantasy'),'Mean'].values[0],'fantasytotalliking':subMostLiked.loc[(subMostrated['Categories'] == 'Books') & (subMostrated['subCategory'] == 'fantasy'),'Countlike'].values[0]})
    else:
        info.append({'fantasytotalrating':0,'fantasyrated':0,'fantasytotalliking':0})
    if (((dfratinguser.Categories == 'Books') & (dfratinguser.subCategory == 'general')).any()) == True and (((dflikeuser.Categories == 'Books') & (dflikeuser.subCategory == 'general')).any()) == True:
        info.append({'generaltotalrating':subMostrated.loc[(subMostrated['Categories'] == 'Books') & (subMostrated['subCategory'] == 'general'),'Count'].values[0],'generalrated':subMostrated.loc[(subMostrated['Categories'] == 'Books') & (subMostrated['subCategory'] == 'general'),'Mean'].values[0],'generaltotalliking':subMostLiked.loc[(subMostrated['Categories'] == 'Books') & (subMostrated['subCategory'] == 'general'),'Countlike'].values[0]})
    else:
        info.append({'generaltotalrating':0,'generalrated':0,'generaltotalliking':0})
    if (((dfratinguser.Categories == 'Books') & (dfratinguser.subCategory == 'literature')).any()) == True and (((dflikeuser.Categories == 'Books') & (dflikeuser.subCategory == 'literature')).any()) == True:
        info.append({'literaturetotalrating':subMostrated.loc[(subMostrated['Categories'] == 'Books') & (subMostrated['subCategory'] == 'literature'),'Count'].values[0],'literaturerated':subMostrated.loc[(subMostrated['Categories'] == 'Books') & (subMostrated['subCategory'] == 'literature'),'Mean'].values[0],'literaturetotalliking':subMostLiked.loc[(subMostrated['Categories'] == 'Books') & (subMostrated['subCategory'] == 'literature'),'Countlike'].values[0]})
    else:
        info.append({'literaturetotalrating':0,'literaturerated':0,'literaturetotalliking':0})
    if (((dfratinguser.Categories == 'Books') & (dfratinguser.subCategory == 'others')).any()) == True and (((dflikeuser.Categories == 'Books') & (dflikeuser.subCategory == 'others')).any()) == True:
        info.append({'otherstotalrating':subMostrated.loc[(subMostrated['Categories'] == 'Books') & (subMostrated['subCategory'] == 'others'),'Count'].values[0],'othersrated':subMostrated.loc[(subMostrated['Categories'] == 'Books') & (subMostrated['subCategory'] == 'others'),'Mean'].values[0],'otherstotalliking':subMostLiked.loc[(subMostrated['Categories'] == 'Books') & (subMostrated['subCategory'] == 'others'),'Countlike'].values[0]})
    else:
        info.append({'otherstotalrating':0,'othersrated':0,'otherstotalliking':0})
    if (((dfratinguser.Categories == 'Books') & (dfratinguser.subCategory == 'romance')).any()) == True and (((dflikeuser.Categories == 'Books') & (dflikeuser.subCategory == 'romance')).any()) == True:
        info.append({'romancetotalrating':subMostrated.loc[(subMostrated['Categories'] == 'Books') & (subMostrated['subCategory'] == 'romance'),'Count'].values[0],'romancerated':subMostrated.loc[(subMostrated['Categories'] == 'Books') & (subMostrated['subCategory'] == 'romance'),'Mean'].values[0],'romancetotalliking':subMostLiked.loc[(subMostrated['Categories'] == 'Books') & (subMostrated['subCategory'] == 'romance'),'Countlike'].values[0]})
    else:
        info.append({'romancetotalrating':0,'romancerated':0,'romancetotalliking':0})
    if (((dfratinguser.Categories == 'Books') & (dfratinguser.subCategory == 'science fiction')).any()) == True and (((dflikeuser.Categories == 'Books') & (dflikeuser.subCategory == 'science fiction')).any()) == True:
        info.append({'science_fictiontotalrating':subMostrated.loc[(subMostrated['Categories'] == 'Books') & (subMostrated['subCategory'] == 'science fiction'),'Count'].values[0],'science_fictionrated':subMostrated.loc[(subMostrated['Categories'] == 'Books') & (subMostrated['subCategory'] == 'science fiction'),'Mean'].values[0],'science_fictiontotalliking':subMostLiked.loc[(subMostrated['Categories'] == 'Books') & (subMostrated['subCategory'] == 'science fiction'),'Countlike'].values[0]})
    else:
        info.append({'science_fictiontotalrating':0,'science_fictionrated':0,'science_fictiontotalliking':0})
    if (((dfratinguser.Categories == 'Books') & (dfratinguser.subCategory == 'short stories')).any()) == True and (((dflikeuser.Categories == 'Books') & (dflikeuser.subCategory == 'short stories')).any()) == True:
        info.append({'short_storiestotalrating':subMostrated.loc[(subMostrated['Categories'] == 'Books') & (subMostrated['subCategory'] == 'science fiction'),'Count'].values[0],'short_storiesrated':subMostrated.loc[(subMostrated['Categories'] == 'Books') & (subMostrated['subCategory'] == 'science fiction'),'Mean'].values[0],'short_storiestotalliking':subMostLiked.loc[(subMostrated['Categories'] == 'Books') & (subMostrated['subCategory'] == 'science fiction'),'Countlike'].values[0]})
    else:
        info.append({'short_storiestotalrating':0,'short_storiesrated':0,'short_storiestotalliking':0})
    if (((dfratinguser.Categories == 'Books') & (dfratinguser.subCategory == 'young adults')).any()) == True and (((dflikeuser.Categories == 'Books') & (dflikeuser.subCategory == 'young adults')).any()) == True:
        info.append({'young_adultstotalrating':subMostrated.loc[(subMostrated['Categories'] == 'Books') & (subMostrated['subCategory'] == 'young adults'),'Count'].values[0],'young_adultsrated':subMostrated.loc[(subMostrated['Categories'] == 'Books') & (subMostrated['subCategory'] == 'young adults'),'Mean'].values[0],'young_adultstotalliking':subMostLiked.loc[(subMostrated['Categories'] == 'Books') & (subMostrated['subCategory'] == 'young adults'),'Countlike'].values[0]})
    else:
        info.append({'young_adultstotalrating':0,'young_adultsrated':0,'young_adultstotalliking':0})
    if (((dfratinguser.Categories == 'Footwear') & (dfratinguser.subCategory == 'flip flops')).any()) == True and (((dflikeuser.Categories == 'Footwear') & (dflikeuser.subCategory == 'flip flops')).any()) == True:
        info.append({'flip_flopstotalrating':subMostrated.loc[(subMostrated['Categories'] == 'Footwear') & (subMostrated['subCategory'] == 'flip flops'),'Count'].values[0],'flip_flopsrated':subMostrated.loc[(subMostrated['Categories'] == 'Footwear') & (subMostrated['subCategory'] == 'flip flops'),'Mean'].values[0],'flip_flopstotalliking':subMostLiked.loc[(subMostrated['Categories'] == 'Footwear') & (subMostrated['subCategory'] == 'flip flops'),'Countlike'].values[0]})
    else:
        info.append({'flip_flopstotalrating':0,'flip_flopsrated':0,'flip_flopstotalliking':0})
    if (((dfratinguser.Categories == 'Footwear') & (dfratinguser.subCategory == 'sandal')).any()) == True and (((dflikeuser.Categories == 'Footwear') & (dflikeuser.subCategory == 'sandal')).any()) == True:
        info.append({'sandaltotalrating':subMostrated.loc[(subMostrated['Categories'] == 'Footwear') & (subMostrated['subCategory'] == 'sandal'),'Count'].values[0],'sandalrated':subMostrated.loc[(subMostrated['Categories'] == 'Footwear') & (subMostrated['subCategory'] == 'sandal'),'Mean'].values[0],'sandaltotalliking':subMostLiked.loc[(subMostrated['Categories'] == 'Footwear') & (subMostrated['subCategory'] == 'sandal'),'Countlike'].values[0]})
    else:
        info.append({'sandaltotalrating':0,'sandalrated':0,'sandaltotalliking':0})
    if (((dfratinguser.Categories == 'Footwear') & (dfratinguser.subCategory == 'shoe accessories')).any()) == True and (((dflikeuser.Categories == 'Footwear') & (dflikeuser.subCategory == 'shoe accessories')).any()) == True:
        info.append({'shoe_accessoriestotalrating':subMostrated.loc[(subMostrated['Categories'] == 'Footwear') & (subMostrated['subCategory'] == 'shoe accessories'),'Count'].values[0],'shoe_accessoriesrated':subMostrated.loc[(subMostrated['Categories'] == 'Footwear') & (subMostrated['subCategory'] == 'shoe accessories'),'Mean'].values[0],'shoe_accessoriestotalliking':subMostLiked.loc[(subMostrated['Categories'] == 'Footwear') & (subMostrated['subCategory'] == 'shoe accessories'),'Countlike'].values[0]})
    else:
        info.append({'shoe_accessoriestotalrating':0,'shoe_accessoriesrated':0,'shoe_accessoriestotalliking':0})
    if (((dfratinguser.Categories == 'Footwear') & (dfratinguser.subCategory == 'shoes')).any()) == True and (((dflikeuser.Categories == 'Footwear') & (dflikeuser.subCategory == 'shoes')).any()) == True:
        info.append({'shoestotalrating':subMostrated.loc[(subMostrated['Categories'] == 'Footwear') & (subMostrated['subCategory'] == 'shoes'),'Count'].values[0],'shoesrated':subMostrated.loc[(subMostrated['Categories'] == 'Footwear') & (subMostrated['subCategory'] == 'shoes'),'Mean'].values[0],'shoestotalliking':subMostLiked.loc[(subMostrated['Categories'] == 'Footwear') & (subMostrated['subCategory'] == 'shoes'),'Countlike'].values[0]})
    else:
        info.append({'shoestotalrating':0,'shoesrated':0,'shoestotalliking':0})
    if (((dfratinguser.Categories == 'Footwear') & (dfratinguser.subCategory == 'socks')).any()) == True and (((dflikeuser.Categories == 'Footwear') & (dflikeuser.subCategory == 'socks')).any()) == True:
        info.append({'sockstotalrating':subMostrated.loc[(subMostrated['Categories'] == 'Footwear') & (subMostrated['subCategory'] == 'socks'),'Count'].values[0],'socksrated':subMostrated.loc[(subMostrated['Categories'] == 'Footwear') & (subMostrated['subCategory'] == 'socks'),'Mean'].values[0],'sockstotalliking':subMostLiked.loc[(subMostrated['Categories'] == 'Footwear') & (subMostrated['subCategory'] == 'socks'),'Countlike'].values[0]})
    else:
        info.append({'sockstotalrating':0,'socksrated':0,'sockstotalliking':0})
    if (((dfratinguser.Categories == 'Personal Care') & (dfratinguser.subCategory == 'bath and body')).any()) == True and (((dflikeuser.Categories == 'Personal Care') & (dflikeuser.subCategory == 'bath and body')).any()) == True:
        info.append({'bath_bodytotalrating':subMostrated.loc[(subMostrated['Categories'] == 'Personal Care') & (subMostrated['subCategory'] == 'bath and body'),'Count'].values[0],'bath_bodyrated':subMostrated.loc[(subMostrated['Categories'] == 'Personal Care') & (subMostrated['subCategory'] == 'bath and body'),'Mean'].values[0],'bath_bodytotalliking':subMostLiked.loc[(subMostrated['Categories'] == 'Personal Care') & (subMostrated['subCategory'] == 'bath and body'),'Countlike'].values[0]})
    else:
        info.append({'bath_bodytotalrating':0,'bath_bodyrated':0,'bath_bodytotalliking':0})
    if (((dfratinguser.Categories == 'Personal Care') & (dfratinguser.subCategory == 'eyes')).any()) == True and (((dflikeuser.Categories == 'Personal Care') & (dflikeuser.subCategory == 'eyes')).any()) == True:
        info.append({'eyestotalrating':subMostrated.loc[(subMostrated['Categories'] == 'Personal Care') & (subMostrated['subCategory'] == 'eyes'),'Count'].values[0],'eyesrated':subMostrated.loc[(subMostrated['Categories'] == 'Personal Care') & (subMostrated['subCategory'] == 'eyes'),'Mean'].values[0],'eyestotalliking':subMostLiked.loc[(subMostrated['Categories'] == 'Personal Care') & (subMostrated['subCategory'] == 'eyes'),'Countlike'].values[0]})
    else:
        info.append({'eyestotalrating':0,'eyesrated':0,'eyestotalliking':0})
    if (((dfratinguser.Categories == 'Personal Care') & (dfratinguser.subCategory == 'fragrance')).any()) == True and (((dflikeuser.Categories == 'Personal Care') & (dflikeuser.subCategory == 'fragrance')).any()) == True:
        info.append({'fragrancetotalrating':subMostrated.loc[(subMostrated['Categories'] == 'Personal Care') & (subMostrated['subCategory'] == 'fragrance'),'Count'].values[0],'fragrancerated':subMostrated.loc[(subMostrated['Categories'] == 'Personal Care') & (subMostrated['subCategory'] == 'fragrance'),'Mean'].values[0],'fragrancetotalliking':subMostLiked.loc[(subMostrated['Categories'] == 'Personal Care') & (subMostrated['subCategory'] == 'fragrance'),'Countlike'].values[0]})
    else:
        info.append({'fragrancetotalrating':0,'fragrancerated':0,'fragrancetotalliking':0})
    if (((dfratinguser.Categories == 'Personal Care') & (dfratinguser.subCategory == 'hair')).any()) == True and (((dflikeuser.Categories == 'Personal Care') & (dflikeuser.subCategory == 'hair')).any()) == True:
        info.append({'hairtotalrating':subMostrated.loc[(subMostrated['Categories'] == 'Personal Care') & (subMostrated['subCategory'] == 'hair'),'Count'].values[0],'hairrated':subMostrated.loc[(subMostrated['Categories'] == 'Personal Care') & (subMostrated['subCategory'] == 'hair'),'Mean'].values[0],'hairtotalliking':subMostLiked.loc[(subMostrated['Categories'] == 'Personal Care') & (subMostrated['subCategory'] == 'hair'),'Countlike'].values[0]})
    else:
        info.append({'hairtotalrating':0,'hairrated':0,'hairtotalliking':0})
    if (((dfratinguser.Categories == 'Personal Care') & (dfratinguser.subCategory == 'lips')).any()) == True and (((dflikeuser.Categories == 'Personal Care') & (dflikeuser.subCategory == 'lips')).any()) == True:
        info.append({'lipstotalrating':subMostrated.loc[(subMostrated['Categories'] == 'Personal Care') & (subMostrated['subCategory'] == 'lips'),'Count'].values[0],'lipsrated':subMostrated.loc[(subMostrated['Categories'] == 'Personal Care') & (subMostrated['subCategory'] == 'lips'),'Mean'].values[0],'lipstotalliking':subMostLiked.loc[(subMostrated['Categories'] == 'Personal Care') & (subMostrated['subCategory'] == 'lips'),'Countlike'].values[0]})
    else:
        info.append({'lipstotalrating':0,'lipsrated':0,'lipstotalliking':0})
    if (((dfratinguser.Categories == 'Personal Care') & (dfratinguser.subCategory == 'makeup')).any()) == True and (((dflikeuser.Categories == 'Personal Care') & (dflikeuser.subCategory == 'makeup')).any()) == True:
        info.append({'makeuptotalrating':subMostrated.loc[(subMostrated['Categories'] == 'Personal Care') & (subMostrated['subCategory'] == 'makeup'),'Count'].values[0],'makeuprated':subMostrated.loc[(subMostrated['Categories'] == 'Personal Care') & (subMostrated['subCategory'] == 'makeup'),'Mean'].values[0],'makeuptotalliking':subMostLiked.loc[(subMostrated['Categories'] == 'Personal Care') & (subMostrated['subCategory'] == 'makeup'),'Countlike'].values[0]})
    else:
        info.append({'makeuptotalrating':0,'makeuprated':0,'makeuptotalliking':0})
    if (((dfratinguser.Categories == 'Personal Care') & (dfratinguser.subCategory == 'nails')).any()) == True and (((dflikeuser.Categories == 'Personal Care') & (dflikeuser.subCategory == 'nails')).any()) == True:
        info.append({'nailstotalrating':subMostrated.loc[(subMostrated['Categories'] == 'Personal Care') & (subMostrated['subCategory'] == 'nails'),'Count'].values[0],'nailsrated':subMostrated.loc[(subMostrated['Categories'] == 'Personal Care') & (subMostrated['subCategory'] == 'nails'),'Mean'].values[0],'nailstotalliking':subMostLiked.loc[(subMostrated['Categories'] == 'Personal Care') & (subMostrated['subCategory'] == 'nails'),'Countlike'].values[0]})
    else:
        info.append({'nailstotalrating':0,'nailsrated':0,'nailstotalliking':0})
    if (((dfratinguser.Categories == 'Personal Care') & (dfratinguser.subCategory == 'skin care')).any()) == True and (((dflikeuser.Categories == 'Personal Care') & (dflikeuser.subCategory == 'skin care')).any()) == True:
        info.append({'skin_caretotalrating':subMostrated.loc[(subMostrated['Categories'] == 'Personal Care') & (subMostrated['subCategory'] == 'skin care'),'Count'].values[0],'skin_carerated':subMostrated.loc[(subMostrated['Categories'] == 'Personal Care') & (subMostrated['subCategory'] == 'skin care'),'Mean'].values[0],'skin_caretotalliking':subMostLiked.loc[(subMostrated['Categories'] == 'Personal Care') & (subMostrated['subCategory'] == 'skin care'),'Countlike'].values[0]})
    else:
       info.append({'skin_caretotalrating':0,'skin_carerated':0,'skin_caretotalliking':0}) 
    if (((dfratinguser.Categories == 'Phones') & (dfratinguser.subCategory == 'android')).any()) == True and (((dflikeuser.Categories == 'Phones') & (dflikeuser.subCategory == 'android')).any()) == True:
        info.append({'androidtotalrating':subMostrated.loc[(subMostrated['Categories'] == 'Phones') & (subMostrated['subCategory'] == 'android'),'Count'].values[0],'androidrated':subMostrated.loc[(subMostrated['Categories'] == 'Phones') & (subMostrated['subCategory'] == 'android'),'Mean'].values[0],'androidtotalliking':subMostLiked.loc[(subMostrated['Categories'] == 'Phones') & (subMostrated['subCategory'] == 'android'),'Countlike'].values[0]})
    else:
        info.append({'androidtotalrating':0,'androidrated':0,'androidtotalliking':0})
    if (((dfratinguser.Categories == 'Phones') & (dfratinguser.subCategory == 'apple')).any()) == True and (((dflikeuser.Categories == 'Phones') & (dflikeuser.subCategory == 'apple')).any()) == True:
        info.append({'appletotalrating':subMostrated.loc[(subMostrated['Categories'] == 'Phones') & (subMostrated['subCategory'] == 'apple'),'Count'].values[0],'applerated':subMostrated.loc[(subMostrated['Categories'] == 'Phones') & (subMostrated['subCategory'] == 'apple'),'Mean'].values[0],'appletotalliking':subMostLiked.loc[(subMostrated['Categories'] == 'Phones') & (subMostrated['subCategory'] == 'apple'),'Countlike'].values[0]})
    else:
        info.append({'appletotalrating':0,'applerated':0,'appletotalliking':0})
    if (((dfratinguser.Categories == 'Phones') & (dfratinguser.subCategory == 'honor')).any()) == True and (((dflikeuser.Categories == 'Phones') & (dflikeuser.subCategory == 'honor')).any()) == True:
        info.append({'honortotalrating':subMostrated.loc[(subMostrated['Categories'] == 'Phones') & (subMostrated['subCategory'] == 'honor'),'Count'].values[0],'honorrated':subMostrated.loc[(subMostrated['Categories'] == 'Phones') & (subMostrated['subCategory'] == 'honor'),'Mean'].values[0],'honortotalliking':subMostLiked.loc[(subMostrated['Categories'] == 'Phones') & (subMostrated['subCategory'] == 'honor'),'Countlike'].values[0]})
    else:
        info.append({'honortotalrating':0,'honorrated':0,'honortotalliking':0})
    if (((dfratinguser.Categories == 'Phones') & (dfratinguser.subCategory == 'huawei')).any()) == True and (((dflikeuser.Categories == 'Phones') & (dflikeuser.subCategory == 'huawei')).any()) == True:
        info.append({'huaweitotalrating':subMostrated.loc[(subMostrated['Categories'] == 'Phones') & (subMostrated['subCategory'] == 'huawei'),'Count'].values[0],'huaweirated':subMostrated.loc[(subMostrated['Categories'] == 'Phones') & (subMostrated['subCategory'] == 'huawei'),'Mean'].values[0],'huaweitotalliking':subMostLiked.loc[(subMostrated['Categories'] == 'Phones') & (subMostrated['subCategory'] == 'huawei'),'Countlike'].values[0]})
    else:
        info.append({'huaweitotalrating':0,'huaweirated':0,'huaweitotalliking':0})
    if (((dfratinguser.Categories == 'Phones') & (dfratinguser.subCategory == 'nokia')).any()) == True and (((dflikeuser.Categories == 'Phones') & (dflikeuser.subCategory == 'nokia')).any()) == True:
        info.append({'nokiatotalrating':subMostrated.loc[(subMostrated['Categories'] == 'Phones') & (subMostrated['subCategory'] == 'nokia'),'Count'].values[0],'nokiarated':subMostrated.loc[(subMostrated['Categories'] == 'Phones') & (subMostrated['subCategory'] == 'nokia'),'Mean'].values[0],'nokiatotalliking':subMostLiked.loc[(subMostrated['Categories'] == 'Phones') & (subMostrated['subCategory'] == 'nokia'),'Countlike'].values[0]})
    else:
        info.append({'nokiatotalrating':0,'nokiarated':0,'nokiatotalliking':0})
    if (((dfratinguser.Categories == 'Phones') & (dfratinguser.subCategory == 'oneplus')).any()) == True and (((dflikeuser.Categories == 'Phones') & (dflikeuser.subCategory == 'oneplus')).any()) == True:
        info.append({'oneplustotalrating':subMostrated.loc[(subMostrated['Categories'] == 'Phones') & (subMostrated['subCategory'] == 'oneplus'),'Count'].values[0],'oneplusrated':subMostrated.loc[(subMostrated['Categories'] == 'Phones') & (subMostrated['subCategory'] == 'oneplus'),'Mean'].values[0],'oneplustotalliking':subMostLiked.loc[(subMostrated['Categories'] == 'Phones') & (subMostrated['subCategory'] == 'oneplus'),'Countlike'].values[0]})
    else:
        info.append({'oneplustotalrating':0,'oneplusrated':0,'oneplustotalliking':0})
    if (((dfratinguser.Categories == 'Phones') & (dfratinguser.subCategory == 'oppo')).any()) == True and (((dflikeuser.Categories == 'Phones') & (dflikeuser.subCategory == 'oppo')).any()) == True:
        info.append({'oppototalrating':subMostrated.loc[(subMostrated['Categories'] == 'Phones') & (subMostrated['subCategory'] == 'oppo'),'Count'].values[0],'opporated':subMostrated.loc[(subMostrated['Categories'] == 'Phones') & (subMostrated['subCategory'] == 'oppo'),'Mean'].values[0],'oppototalliking':subMostLiked.loc[(subMostrated['Categories'] == 'Phones') & (subMostrated['subCategory'] == 'oppo'),'Countlike'].values[0]})
    else:
        info.append({'oppototalrating':0,'opporated':0,'oppototalliking':0})
    if (((dfratinguser.Categories == 'Phones') & (dfratinguser.subCategory == 'redmi')).any()) == True and (((dflikeuser.Categories == 'Phones') & (dflikeuser.subCategory == 'redmi')).any()) == True:
        info.append({'redmitotalrating':subMostrated.loc[(subMostrated['Categories'] == 'Phones') & (subMostrated['subCategory'] == 'redmi'),'Count'].values[0],'redmirated':subMostrated.loc[(subMostrated['Categories'] == 'Phones') & (subMostrated['subCategory'] == 'redmi'),'Mean'].values[0],'redmitotalliking':subMostLiked.loc[(subMostrated['Categories'] == 'Phones') & (subMostrated['subCategory'] == 'redmi'),'Countlike'].values[0]})
    else:
        info.append({'redmitotalrating':0,'redmirated':0,'redmitotalliking':0})
    if (((dfratinguser.Categories == 'Phones') & (dfratinguser.subCategory == 'samsung')).any()) == True and (((dflikeuser.Categories == 'Phones') & (dflikeuser.subCategory == 'samsung')).any()) == True:
        info.append({'samsungtotalrating':subMostrated.loc[(subMostrated['Categories'] == 'Phones') & (subMostrated['subCategory'] == 'samsung'),'Count'].values[0],'samsungrated':subMostrated.loc[(subMostrated['Categories'] == 'Phones') & (subMostrated['subCategory'] == 'samsung'),'Mean'].values[0],'samsungtotalliking':subMostLiked.loc[(subMostrated['Categories'] == 'Phones') & (subMostrated['subCategory'] == 'samsung'),'Countlike'].values[0]})
    else:
        info.append({'samsungtotalrating':0,'samsungrated':0,'samsungtotalliking':0})
    if (((dfratinguser.Categories == 'Phones') & (dfratinguser.subCategory == 'sony')).any()) == True and (((dflikeuser.Categories == 'Phones') & (dflikeuser.subCategory == 'sony')).any()) == True:
        info.append({'sonytotalrating':subMostrated.loc[(subMostrated['Categories'] == 'Phones') & (subMostrated['subCategory'] == 'sony'),'Count'].values[0],'sonyrated':subMostrated.loc[(subMostrated['Categories'] == 'Phones') & (subMostrated['subCategory'] == 'sony'),'Mean'].values[0],'sonytotalliking':subMostLiked.loc[(subMostrated['Categories'] == 'Phones') & (subMostrated['subCategory'] == 'sony'),'Countlike'].values[0]})
    else:
        info.append({'sonytotalrating':0,'sonyrated':0,'sonytotalliking':0})
    if (((dfratinguser.Categories == 'Phones') & (dfratinguser.subCategory == 'vivo')).any()) == True and (((dflikeuser.Categories == 'Phones') & (dflikeuser.subCategory == 'vivo')).any()) == True:    
        info.append({'vivototalrating':subMostrated.loc[(subMostrated['Categories'] == 'Phones') & (subMostrated['subCategory'] == 'vivo'),'Count'].values[0],'vivorated':subMostrated.loc[(subMostrated['Categories'] == 'Phones') & (subMostrated['subCategory'] == 'vivo'),'Mean'].values[0],'vivototalliking':subMostLiked.loc[(subMostrated['Categories'] == 'Phones') & (subMostrated['subCategory'] == 'vivo'),'Countlike'].values[0]})
    else:
        info.append({'vivototalrating':0,'vivorated':0,'vivototalliking':0})
    if (((dfratinguser.Categories == 'Sporting Goods') & (dfratinguser.subCategory == 'bags')).any()) == True and (((dflikeuser.Categories == 'Sporting Goods') & (dflikeuser.subCategory == 'bags')).any()) == True:
        info.append({'bags_Sportingtotalrating':subMostrated.loc[(subMostrated['Categories'] == 'Sporting Goods') & (subMostrated['subCategory'] == 'bags'),'Count'].values[0],'bags_Sportingrated':subMostrated.loc[(subMostrated['Categories'] == 'Sporting Goods') & (subMostrated['subCategory'] == 'bags'),'Mean'].values[0],'bags_Sportingtotalliking':subMostLiked.loc[(subMostrated['Categories'] == 'Sporting Goods') & (subMostrated['subCategory'] == 'bags'),'Countlike'].values[0]})
    else:
        info.append({'bags_Sportingtotalrating':0,'bags_Sportingrated':0,'bags_Sportingtotalliking':0})
    if (((dfratinguser.Categories == 'Sporting Goods') & (dfratinguser.subCategory == 'eyewear')).any()) == True and (((dflikeuser.Categories == 'Sporting Goods') & (dflikeuser.subCategory == 'eyewear')).any()) == True:
        info.append({'eyewear_Sportingtotalrating':subMostrated.loc[(subMostrated['Categories'] == 'Sporting Goods') & (subMostrated['subCategory'] == 'eyewear'),'Count'].values[0],'eyewear_Sportingrated':subMostrated.loc[(subMostrated['Categories'] == 'Sporting Goods') & (subMostrated['subCategory'] == 'eyewear'),'Mean'].values[0],'eyewear_Sportingtotalliking':subMostLiked.loc[(subMostrated['Categories'] == 'Sporting Goods') & (subMostrated['subCategory'] == 'eyewear'),'Countlike'].values[0]})
    else:
        info.append({'eyewear_Sportingtotalrating':0,'eyewear_Sportingrated':0,'eyewear_Sportingtotalliking':0})
    if (((dfratinguser.Categories == 'Sporting Goods') & (dfratinguser.subCategory == 'headwear')).any()) == True and (((dflikeuser.Categories == 'Sporting Goods') & (dflikeuser.subCategory == 'headwear')).any()) == True:
        info.append({'headwear_Sportingtotalrating':subMostrated.loc[(subMostrated['Categories'] == 'Sporting Goods') & (subMostrated['subCategory'] == 'headwear'),'Count'].values[0],'headwear_Sportingrated':subMostrated.loc[(subMostrated['Categories'] == 'Sporting Goods') & (subMostrated['subCategory'] == 'headwear'),'Mean'].values[0],'headwear_Sportingtotalliking':subMostLiked.loc[(subMostrated['Categories'] == 'Sporting Goods') & (subMostrated['subCategory'] == 'headwear'),'Countlike'].values[0]})
    else:
        info.append({'headwear_Sportingtotalrating':0,'headwear_Sportingrated':0,'headwear_Sportingtotalliking':0})
    if (((dfratinguser.Categories == 'Sporting Goods') & (dfratinguser.subCategory == 'sports equipment')).any()) == True and (((dflikeuser.Categories == 'Sporting Goods') & (dflikeuser.subCategory == 'sports equipment')).any()) == True:
        info.append({'sports_equipmenttotalrating':subMostrated.loc[(subMostrated['Categories'] == 'Sporting Goods') & (subMostrated['subCategory'] == 'sports equipment'),'Count'].values[0],'sports_equipmentrated':subMostrated.loc[(subMostrated['Categories'] == 'Sporting Goods') & (subMostrated['subCategory'] == 'sports equipment'),'Mean'].values[0],'sports_equipmenttotalliking':subMostLiked.loc[(subMostrated['Categories'] == 'Sporting Goods') & (subMostrated['subCategory'] == 'sports equipment'),'Countlike'].values[0]})
    else:
        info.append({'sports_equipmenttotalrating':0,'sports_equipmentrated':0,'sports_equipmenttotalliking':0})
    if (((dfratinguser.Categories == 'Sporting Goods') & (dfratinguser.subCategory == 'watches')).any()) == True and (((dflikeuser.Categories == 'Sporting Goods') & (dflikeuser.subCategory == 'watches')).any()) == True:
        info.append({'watches_Sportingtotalrating':subMostrated.loc[(subMostrated['Categories'] == 'Sporting Goods') & (subMostrated['subCategory'] == 'watches'),'Count'].values[0],'watches_Sportingrated':subMostrated.loc[(subMostrated['Categories'] == 'Sporting Goods') & (subMostrated['subCategory'] == 'watches'),'Mean'].values[0],'watches_Sportingtotalliking':subMostLiked.loc[(subMostrated['Categories'] == 'Sporting Goods') & (subMostrated['subCategory'] == 'watches'),'Countlike'].values[0]})
    else:
        info.append({'watches_Sportingtotalrating':0,'watches_Sportingrated':0,'watches_Sportingtotalliking':0})
    if (((dfratinguser.Categories == 'Sporting Goods') & (dfratinguser.subCategory == 'water bottle')).any()) == True and (((dflikeuser.Categories == 'Sporting Goods') & (dflikeuser.subCategory == 'water bottle')).any()) == True:
        info.append({'water_bottletotalrating':subMostrated.loc[(subMostrated['Categories'] == 'Sporting Goods') & (subMostrated['subCategory'] == 'water bottle'),'Count'].values[0],'water_bottlerated':subMostrated.loc[(subMostrated['Categories'] == 'Sporting Goods') & (subMostrated['subCategory'] == 'water bottle'),'Mean'].values[0],'water_bottletotalliking':subMostLiked.loc[(subMostrated['Categories'] == 'Sporting Goods') & (subMostrated['subCategory'] == 'water bottle'),'Countlike'].values[0]})
    else:
        info.append({'water_bottletotalrating':0,'water_bottlerated':0,'water_bottletotalliking':0})
    
    return jsonify(info)

if __name__ == '__main__':
    app.run(debug=False)