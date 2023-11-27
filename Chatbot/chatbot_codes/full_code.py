import keras
import nltk
import pickle
import json
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation
from flask_mysqldb import MySQL
import mysql.connector
import random
from pygame import mixer
import pymysql
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

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
        
        data = pd.read_csv("C:/Users/HU/Desktop/Chatbot/chatbot_codes/breast_cancer_symptoms.csv")

        # Modify the dataset based on the given parameters
        data = data[["Lump in the breast", "Thickening or swelling in the breast", "Change in the size or shape of the breast", "Dimpling or puckering of the skin on the breast", "Inverted nipple", "Redness or scaling on the breast or nipple", "Nipple discharge", "Swollen lymph nodes under the arm", "severity_level"]]

        # Split the dataset into input features and target variable
        X = data.drop("severity_level", axis=1)
        y = data["severity_level"]

        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Train a decision tree model on the training set
        model = DecisionTreeClassifier()
        weights2 = model.fit(X_train, y_train)

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

        
    if tag == 'healthy lifestyle':
        print("Maintaining a healthy lifestyle is important for overall health and wellbeing. Some tips for a healthy lifestyle include eating a balanced diet, getting regular exercise, getting enough sleep, managing stress, and avoiding unhealthy habits such as smoking or excessive alcohol consumption. How would you like to improve your health?")
        response = input()
        habits = []
        if "diet" in response:
            print("That's great. What are some foods that you currently enjoy eating?")
            foods = input().split(", ")
            print("It's important to incorporate a variety of foods into your diet. Here are some healthy meal ideas based on the foods you mentioned:")
            
            for food in foods:
                print("- Grilled", food, "with a side of roasted vegetables")
                print("- A salad with", food, ", mixed greens, and a vinaigrette dressing")
                print("- A smoothie with", food, ", Greek yogurt, and spinach")
            
        elif "exercise" in response:
            print("Good choice. What are some physical activities that you enjoy doing?")
            activities = input().split(", ")
            print("It's important to get at least 150 minutes of moderate-intensity aerobic exercise per week. Here are some physical activities you might enjoy:")
            
            for activity in activities:
                print("-", activity, "for 30 minutes, 5 times per week")
                print("- Joining a local sports team or fitness class that focuses on", activity)
            
        elif "sleep" in response:
                print("Sleep is important for overall health. How many hours of sleep are you currently getting?")
                hours = int(input())

                if hours >= 7:
                 print("That's great. Keep up the good work!")
                else:
                  print("It's important to aim for at least 7 hours of sleep per night. Here are some tips for improving your sleep:")
                  print("- Establishing a regular sleep schedule and sticking to it")
                  print("- Avoiding caffeine and alcohol before bedtime")
                  print("- Creating a relaxing bedtime routine, such as taking a warm bath or reading a book")
        elif "stress" in response:
            print("Stress can have negative effects on your health. What are some activities that you find relaxing?")
            activities = input().split(", ")
            print("It's important to make time for relaxation and stress management. Here are some activities you might find helpful:")
            
            for activity in activities:
                print("- Practicing mindfulness meditation or deep breathing exercises")
                print("- Taking a yoga or tai chi class that focuses on relaxation")
                print("- Engaging in a hobby or activity that you enjoy, such as reading or listening to music")

        elif "unhealthy habits" in response:
             print("Unhealthy habits can have negative effects on your health. What are some habits that you would like to change?")
             habits = input().split(", ")
             print("It's important to take small steps to change unhealthy habits over time. Here are some tips for breaking bad habits:")
        
        for habit in habits:
            print("- Setting a specific goal for changing the habit, such as cutting back on smoking or drinking")
            print("- Identifying triggers that lead to the habit and finding ways to avoid or cope with them")
            print("- Enlisting the support of friends or family members who can help you stay on track")    
        
    if tag=='hospital_recommendation':
            response = "Sure, I can recommend some hospitals based on your preferences. Do you prefer private or public hospitals?"
            return response, tag
    if tag=='private_hospital_recommendation':
        # Fetch hospitals from database based on rating
        cur = mysql.cursor()
        cur.execute("SELECT hname, rating FROM facility WHERE hospital_type = 'private' ORDER BY rating DESC")
        results = cur.fetchall()
        if len(results) == 0:
            response = 'Sorry, we could not find any private hospitals for breast cancer treatment.'
        else:
            response = 'Here are some private hospitals that we recommend for breast cancer treatment:\n'
            for result in results:
                response += f'{result[0]} - {result[1]}\n'
        return response, tag
    if tag=="vital_signs":
        print("Vital signs are measurements of the body's basic functions, including body temperature, blood pressure, pulse rate, and respiratory rate. Which vital sign do want to measure.")
        response = input()
       
        if "temprature" in response:
          print ("do you have any measument device to record vital signs")
          res = input()
          if "yes" in res: 
             print("Great! Let's start with the temperature.")
             print("measure the the body temprature and enter the result")

             temperature = float(input())
             if  temperature > 37.5:
              print("According to your response you have a fever.")
             else :
              print("You do not have a fever.")  
          elif "no" in res:
              print("To measure your body temperature, \nyou can use a regular thermometer that you place under your tongue.") 
              print("Enter the response you see on the thermometer")
              temperature = float(input())
              if  temperature > 37.5:
                  print("According to your response you have a fever.")
              else :
                  print("You do not have a fever.")  
        elif "blood pressure" in response: 
    
            print("OK, let's measure , blood pressure.")
            print ("do you have any measurement device to record the vital sign")
            res = input()
            if "yes" in res: 
               print("Great! Let's start with the blood pressure.")
               
               systolic = int(input("Please enter the patient's systolic blood pressure: "))
               diastolic = int(input("Please enter the patient's diastolic blood pressure: "))
               if systolic >= 140 or diastolic >= 90:
                  print("It looks like your blood pressure is above normal \n we recommend you seek medical attention")
               else:
                  print("Congrats! your blood pressure is normal.")
            elif "no" in res:
                print("If you do not have a blood pressure monitor, it may be difficult to measure your blood pressure accurately.")
                print("One way is to check for symptoms that may indicate high or low blood pressure. \nFor example, if you feel dizzy, lightheaded, or have a headache, it may be a sign of high blood pressure. \nIf you feel weak or fatigued, it may be a sign of low blood pressure.")
                print("Enter what you got down below to check if it's normal")
        elif "respiratory rate" in response: 
            print("OK, let's measure your repiratory rate.")
            print("To measure your respiratory rate, \nyou can count number of breaths you take in 30 seconds and multiply by two.")
            rate = int(input())
            if rate > 20 & rate < 12:
               print("Your respiratory rate is outside of the normal range, \nit's important to consult with a healthcare provider.")
            else:
                print("Your respiratory rate looks good")
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

'''
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
'''