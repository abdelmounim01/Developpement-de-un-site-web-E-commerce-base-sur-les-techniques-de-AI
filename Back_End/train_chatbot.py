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
data_file = open('C:/Users/abd/OneDrive/Bureau/PFE_FILES_FIN/PFE.json').read()
intents = json.loads(data_file)

#-------------------------------
#        Preprocessing
#-------------------------------
for intent in intents['intents']:
    for pattern in intent['patterns']:

        #tokenize each word
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        #add in documents questions and its tags
        documents.append((w, intent['tag']))

        # add tags in classes
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# lemmaztize , lower and remove punctuation from each word and remove duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in string.punctuation]
words = sorted(list(set(words)))
# sorter classes
classes = sorted(list(set(classes)))
# documents = combination of questions and tags
print (len(documents), "documents")
# classes = tags
print (len(classes), "classes", classes)
# words = all the question words after preprocessing
print (len(words), "unique lemmatized words", words)

#save words and classes
pickle.dump(words,open('C:/Users/abd/OneDrive/Bureau/PFE_FILES_FIN/PFEwords.pkl','wb'))
pickle.dump(classes,open('C:/Users/abd/OneDrive/Bureau/PFE_FILES_FIN/PFEclasses.pkl','wb'))


#-------------------------------
# create our training data
#-------------------------------
 
training = []
# create an empty array for our output
output_empty = [0] * len(classes)
# training, bag of words for each sentence : binary
for doc in documents:
    # initialize our bag of words
    bag = []
    # create list of questions (pattern)
    pattern_words = doc[0]
    # lemmatize each word - create the wordbase
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    # create our bag of words: table takes 1 for words found in our current question and 0 for other words
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
    
    # out for 0 for all tags and 1 for the current tag (current)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    
    training.append([bag, output_row])
    
# shuffle the lines of our training to have different results for each training and change to np.array (matrix)
random.shuffle(training)
training = np.array(training) #pour le rendre sous forme de matrice
# create train x(question) and y(tag). X -pattern(r)=question (bag), Y - tag (output_row)
train_x = list(training[:,0])
train_y = list(training[:,1])
print("Training data created")

#-------------------------------
# create the model
#-------------------------------
#Create the model - 3 layers. The first layer 128 neurons, the second layer 64 neurons and the third output layer
#contain a number of neurons equal to the number of tags to predict the output tag with softmax: in our case 133 tag
model = Sequential() 
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu')) 
model.add(Dropout(0.5)) 
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax')) 

# Compile the model. Stochastic gradient descent with Nesterov accelerated the gradient and gives good results for the model
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True) 
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy']) 

hist = model.fit(np.array(train_x), np.array(train_y), epochs=2150, batch_size=30, verbose=1) 
model.save('C:/Users/abd/OneDrive/Bureau/PFE_FILES_FIN/PFEchatbot_model.h5', hist)

print("model created")
