import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import string
import json
import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import random

words=[]
classes = []
documents = []
data_file = open('C:/Users/errza/OneDrive/Bureau/PFE_FILES_FIN/PFE.json').read()
intents = json.loads(data_file)

for intent in intents['intents']:
    for pattern in intent['patterns']:

        #tokenize Chaque mot
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        #ajouter dans documents questions et ses tag
        documents.append((w, intent['tag']))

        # ajouter les tag dans classes
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# lemmaztize et lower et supprimer ponctuations de chaque mot et supprimer les doublons
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in string.punctuation]
words = sorted(list(set(words)))
# sorter classes
classes = sorted(list(set(classes)))
# documents = combinaison entre questions et tags
print (len(documents), "documents")
# classes = tags
print (len(classes), "classes", classes)
# words = tout les mots de questions aprés prétraitement
print (len(words), "unique lemmatized words", words)

#sauvegarder words et classes
pickle.dump(words,open('C:/Users/errza/OneDrive/Bureau/PFE_FILES_FIN/PFEwords.pkl','wb'))
pickle.dump(classes,open('C:/Users/errza/OneDrive/Bureau/PFE_FILES_FIN/PFEclasses.pkl','wb'))

# créer notre training data
training = []
# créer un tableau vide pour notre output
output_empty = [0] * len(classes)
# training, bag des mots pour chaque phrase(question) : binaire 
for doc in documents:
    # initialiser notre bag des mots
    bag = []
    # créer liste des questions(pattern) prétraiter
    pattern_words = doc[0]
    # lemmatize chaque mot - créer la base de mot
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    # créer notre bag de mots : tableau prend 1 pour mots trouver dans notre question courant et 0 pour les autres mots
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
    
    # out pour 0 pour tout les tags et 1 pour le tag courant(actuelle)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    
    training.append([bag, output_row])
    
# shuffle les lignes de notre training pour avoir des résultat différent pour chaque training et changer en np.array(matrice)
random.shuffle(training)
training = np.array(training) #pour le rendre sous forme de matrice
# create train x(question) et y(tag). X -pattern(r)=question (bag), Y - tag (output_row)
train_x = list(training[:,0])
train_y = list(training[:,1])
print("Training data created")

#Créer un modèle - 3 couches. La première couche 128 neurones, la deuxième couche 64 neurones et la troisième couche de sortie
#contiennent un nombre de neurones égal au nombre des tags pour prédire le tag de sortie avec softmax: dans notre cas 133 tag
model = Sequential() # model de keras faire organiser les couches
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu')) # relu pour rendre les valeurs négative à 0 et pour étre une fonction linéaire
model.add(Dropout(0.5)) # une couche d'abandon ignore un ensemble de neurones (au hasard) , ce qui permet d'éviter le surapprentissage.
model.add(Dense(64, activation='relu')) # Une couche dense est une couche de réseau neuronal classique entièrement connectée: chaque nœud d'entrée est connecté à chaque nœud de sortie.
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax')) # softmax pour faire prédiction de 0 à 1 utilisé pour multiple classes

# Compiler model. Stochastic gradient descent avec Nesterov accéléré le gradient et donne de bons résultats pour le model
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True) # utilisée pour l'optimisation d'une fonction objectif.
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy']) # S'il s'agit d'un problème multiclasse, vous devez utiliser categorical_crossentropy. Les étiquettes doivent également être converties au format catégoriel.
#training(en donne le input) et sauvegarder le model
hist = model.fit(np.array(train_x), np.array(train_y), epochs=2150, batch_size=30, verbose=1) 
model.save('C:/Users/errza/OneDrive/Bureau/PFE_FILES_FIN/PFEchatbot_model.h5', hist)

print("model created")
