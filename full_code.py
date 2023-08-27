
import keras
import nltk
import pickle
import json
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation
import random
import datetime
from googlesearch import *
import webbrowser
import requests
from pycricbuzz import Cricbuzz
import billboard
import time
from pygame import mixer
import COVID19Py
import pymysql
import pandas
import pandas as pd

#ignore_words = ['the', 'a', 'an', 'is', 'are', 'was', 'were', 'in', 'at', 'to', 'on', 'for', 'with', 'of', 'and', 'or', 'that', 'this', 'it', 'not', 'be', 'from', 'as', 'by', 'you', 'me', 'my', 'your', 'our', 'their', 'his', 'her']


mysql = pymysql.connect(host="localhost", user="root", password="", database="crud")

from nltk.stem import WordNetLemmatizer
lemmatizer=WordNetLemmatizer()

words=[]
classes=[]
documents=[]
ignore=['?','!',',',"'s"]

data_file=open('C:/Users/HU/Desktop/Chatbot/chatbot_codes/intents.json').read()
intents=json.loads(data_file)

for intent in intents['intents']:
    for pattern in intent['patterns']:
        w=nltk.word_tokenize(pattern)
        words.extend(w)
        documents.append((w,intent['tag']))
        
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
            
words=[lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore]
words=sorted(list(set(words)))
classes=sorted(list(set(classes)))
pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))

#training data
training=[]
output_empty=[0]*len(classes)

for doc in documents:
    bag=[]
    pattern=doc[0]
    pattern=[ lemmatizer.lemmatize(word.lower()) for word in pattern ]
    
    for word in words:
        if word in pattern:
            bag.append(1)
        else:
            bag.append(0)
    output_row=list(output_empty)
    output_row[classes.index(doc[1])]=1
    
    training.append([bag,output_row])
    
random.shuffle(training)
training=np.array(training)  
X_train=list(training[:,0])
y_train=list(training[:,1])  

#Model
model=Sequential()
model.add(Dense(128,activation='relu',input_shape=(len(X_train[0]),)))
model.add(Dropout(0.5))
model.add(Dense(64,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(y_train[0]),activation='softmax'))

adam=keras.optimizers.Adam(0.001)
model.compile(optimizer=adam,loss='categorical_crossentropy',metrics=['accuracy'])
#model.fit(np.array(X_train),np.array(y_train),epochs=200,batch_size=10,verbose=1)
weights=model.fit(np.array(X_train),np.array(y_train),epochs=200,batch_size=10,verbose=1)    
model.save('mymodel.h5',weights)

from keras.models import load_model
model = load_model('mymodel.h5')
intents = json.loads(open('C:/Users/HU/Desktop/Chatbot/chatbot_codes/intents.json').read())
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))


#Predict
def clean_up(sentence):
    sentence_words=nltk.word_tokenize(sentence)
    sentence_words=[ lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


def create_bow(sentence,words):
    sentence_words=clean_up(sentence)
    bag=list(np.zeros(len(words)))
    
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence,model):
    p=create_bow(sentence,words)
    res=model.predict(np.array([p]))[0]
    threshold=0.8
    results=[[i,r] for i,r in enumerate(res) if r>threshold]
    results.sort(key=lambda x: x[1],reverse=True)
    return_list=[]
    
    for result in results:
        return_list.append({'intent':classes[result[0]],'prob':str(result[1])})
    return return_list

def get_response(return_list,intents_json):
    
    if len(return_list)==0:
        tag='noanswer'
    else:    
        tag=return_list[0]['intent']
    
    if tag == 'not feeling well':
        # Load the dataset
        
        data = pd.read_csv("breast_cancer_symptoms.csv")

        # Modify the dataset based on the given parameters
        data = data[["Lump in the breast", "Thickening or swelling in the breast", "Change in the size or shape of the breast", "Dimpling or puckering of the skin on the breast", "Inverted nipple", "Redness or scaling on the breast or nipple", "Nipple discharge", "Swollen lymph nodes under the arm", "severity_level"]]

        # Split the dataset into input features and target variable
        X = data.drop("severity_level", axis=1)
        y = data["severity_level"]

        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Train a decision tree model on the training set
        model = DecisionTreeClassifier()
        model.fit(X_train, y_train)

        # Evaluate the model on the testing set
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy:", accuracy)

        # Ask the user for each symptom and predict the severity level
        print("Please answer the following questions about your symptoms:")
        severity_mapping = {"Low severity": 1, "Medium severity": 2, "High severity": 3}
        symptoms = []
        for feature in X.columns:
            value = input(f"{feature}: ")
            symptoms.append(int(value))
        symptom_data = pd.DataFrame([symptoms], columns=X.columns)
        severity_levels = model.predict(symptom_data)
        severity_levels = [severity_mapping[level] for level in severity_levels]
        summ = sum(severity_levels)
        # Classify the severity level into 3 categories
        if summ >= 2:
            severity_category = "High severity"
            medical_attention = "We recommend that you seek immediate medical attention."
        elif summ == 1:
            severity_category = "Medium severity"
            medical_attention = "We recommend that you schedule an appointment with your primary care physician."
        else:
            severity_category = "Low severity"
            medical_attention = "We recommend that you monitor your symptoms and seek medical attention if they worsen."

        # Output the final message
    
        
        
        print(f"According to your responses, your severity level is {summ} which is {severity_category}.")
        print(medical_attention)
        
    if tag=='hospital_recommendation':
            response = "Sure, I can recommend some hospitals based on your preferences. Do you prefer private or public hospitals?"
            return response, tag
    if tag=='private_hospital_recommendation':
        # Fetch hospitals from database based on rating
        
        cur.execute("SELECT hname, rating FROM facility WHERE hospital_type = 'private' ORDER BY rating DESC", ('%breast cancer%',))
        results = cur.fetchall()
        if len(results) == 0:
            response = 'Sorry, we could not find any private hospitals for breast cancer treatment.'
        else:
            response = 'Here are some private hospitals that we recommend for breast cancer treatment:\n'
            for result in results:
                response += f'{result[0]} - {result[1]}\n'
        return response, tag
    list_of_intents= intents_json['intents']    
    for i in list_of_intents:
        if tag==i['tag'] :
            result= random.choice(i['responses'])
    return result
    
    if tag=='public_hospital_recommendation':
        # Fetch hospitals from database based on rating
        
        cur.execute("SELECT hname, rating FROM facility WHERE hospital_type = 'public' ORDER BY rating DESC", ('%breast cancer%',))
        results = cur.fetchall()
        if len(results) == 0:
            response = 'Sorry, we could not find any public hospitals for breast cancer treatment.'
        else:
            response = 'Here are some public hospitals that we recommend for breast cancer treatment:\n'
            for result in results:
                response += f'{result[0]} - {result[1]}\n'
        return response, tag
    list_of_intents= intents_json['intents']    
    for i in list_of_intents:
        if tag==i['tag'] :
            result= random.choice(i['responses'])
    return result

from session_management import get_session_dict, update_session_dict

def response(user_input, session_id):
    session_dict = get_session_dict(session_id)
    past_conversation = session_dict.get('past_conversation', [])
    past_conversation.append(user_input)
    update_session_dict(session_id, 'past_conversation', past_conversation)
    
    return_list = predict_class(user_input, model)
    # load the intents object
    response = get_response(return_list, intents)
    
    return response

    
import uuid
while(1):
    x=input()
    print(response(x,uuid))
    if x.lower() in ['bye','goodbye','get lost','see you']:  
        break


#Self learning
print('Help me Learn?')
tag=input('Please enter general category of your question  ')
flag=-1
for i in range(len(intents['intents'])):
    if tag.lower() in intents['intents'][i]['tag']:
        intents['intents'][i]['patterns'].append(input('Enter your message: '))
        intents['intents'][i]['responses'].append(input('Enter expected reply: '))        
        flag=1

if flag==-1:
    
    intents['intents'].append (
        {'tag':tag,
         'patterns': [input('Please enter your message')],
         'responses': [input('Enter expected reply')]})
    
with open('intents.json','w') as outfile:
    outfile.write(json.dumps(intents,indent=4))
