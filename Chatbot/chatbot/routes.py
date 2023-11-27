from chatbot import app
from flask import Flask, render_template, request, url_for, flash, session, jsonify
from chatbot.forms import chatbotform
from chatbot.__init__ import model, words, classes, intents
from werkzeug.utils import redirect
from flask_mysqldb import MySQL
import mysql.connector
import re
import pymysql
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import nltk
import pickle
import json
import numpy as np
from keras.models import Sequential, load_model
import random
import pandas as pd

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()


# Predict
def clean_up(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(
        word.lower()) for word in sentence_words]
    return sentence_words


def create_bow(sentence, words):
    sentence_words = clean_up(sentence)
    bag = list(np.zeros(len(words)))

    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)


def predict_class(sentence, model):
    p = create_bow(sentence, words)
    res = model.predict(np.array([p]))[0]
    threshold = 0.8
    results = [[i, r] for i, r in enumerate(res) if r > threshold]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []

    for result in results:
        return_list.append(
            {'intent': classes[result[0]], 'prob': str(result[1])})
    return return_list


def get_response(return_list, intents_json, text):

    if len(return_list) == 0:
        tag = 'noanswer'
    else:
        tag = return_list[0]['intent']

  
    if tag == 'hospital_recommendation':
            
            response = "Sure, I can recommend some hospitals based on your preferences. Do you prefer private or public hospitals?"
            response, tag

    if tag == 'private_hospital_recommendation':
        # Fetch hospitals from database based on  ('%breast cancer%',))
        
        cur = mysql.cursor()
        cur.execute(
            "SELECT hname, rating, working_hour, severity_levels FROM facility WHERE hospital_type='private' AND severity_levels='High' ORDER BY rating DESC")
        results = cur.fetchall()
        if len(results) == 0:
            response = 'Sorry, we could not find any private hospitals for breast cancer treatment.'
        else:
            response = 'Here are some private hospitals that we recommend for breast cancer treatment:\n'
            for result in results:
                response += f'{result[0]} - {result[1]}\n'
        return response, tag
    if tag == 'public_hospital_recommendation':
        
            # Fetch hospitals from database based on rating
        cur = mysql.cursor()

        cur.execute(
            "SELECT hname, rating, working_hour, severity_levels FROM facility WHERE hospital_type='public' AND severity_levels='High' ORDER BY rating DESC")
        results = cur.fetchall()
        if len(results) == 0:
            response = 'Sorry, we could not find any public hospitals for breast cancer treatment.'
        else:
            response = 'Here are some public hospitals that we recommend for breast cancer treatment:\n'
            for result in results:
                response += f'{result[0]} - {result[1]}\n'
        return response, tag
    
    
    if tag == 'yes':
        
        data = pd.read_csv("C:/Users/HU/Desktop/Chatbot/chatbot_codes/breast_cancer_symptoms.csv")

        # Modify the dataset based on the given parameters
        data = data[["Lump in the breast", "Thickening or swelling in the breast", "Change in the size or shape of the breast", "Dimpling or puckering of the skin on the breast",
            "Inverted nipple", "Redness or scaling on the breast or nipple", "Nipple discharge", "Swollen lymph nodes under the arm", "severity_level"]]

        # Split the dataset into input features and target variable
        X = data.drop("severity_level", axis=1)
        y = data["severity_level"]

        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Train a decision tree model on the training set
        model = DecisionTreeClassifier()
        model.fit(X_train, y_train)
        #model.save('ourmodel.h5',weights2)
        
        # Evaluate the model on the testing set
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy:", accuracy)
        
        symptoms = []
        medical_attention = str()
        for feature in X.columns:
            value = input(f"{feature}: ")
            symptoms.append(int(value))
        symptom_data = pd.DataFrame([symptoms], columns=X.columns)
        severity_levels = model.predict(symptom_data) 
        if "High severity" in severity_levels:
            medical_attention = "We recommend that you seek immediate medical attention based on your preference. Would you prefer a private hospital or a public hospital?"
        elif "Medium severity" in severity_levels:
            medical_attention = "We recommend that you schedule an appointment with your primary care physician."
        else:
            medical_attention = "We recommend that you monitor your symptoms and seek medical attention if they worsen."
            
        return str(severity_levels) + " " " "+ medical_attention, tag

    
    if tag == 'healthy lifestyle':
            response = "Maintaining a healthy lifestyle is important for overall health and wellbeing. Some tips for a healthy lifestyle include eating a balanced diet, getting regular exercise, getting enough sleep, managing stress, and avoiding unhealthy habits such as smoking or excessive alcohol consumption. How would you like to improve your health?"
            return response, tag

    elif tag == 'diet':
             response = "A healthy diet can help maintain a healthy weight and reduce the risk of cancer recurrence. Aim for a diet that is rich in fruits, vegetables, whole grains, and lean protein sources, and limit processed and red meats, sugary drinks, and high-fat food"
             return response, tag 

    elif tag == 'exercise':
             response = " Regular physical activity can help improve overall health and reduce the risk of cancer recurrence. Aim for at least 150 minutes of moderate-intensity exercise per week, such as brisk walking, cycling, or swimming."
             return response, 'exercise'

    elif tag == 'sleep':
             response = " Some strategies that may help improve sleep include: 1:Establishing a regular sleep schedule: Going to bed and waking up at the same time every day can help regulate the sleep-wake cycle.2:Creating a relaxing sleep environment: The sleep environment should be cool, dark, and quiet, and free from distractions such as electronic devices.3.Practicing relaxation techniques: Relaxation techniques such as deep breathing, meditation, or progressive muscle relaxation may help reduce stress and promote sleep.4:Avoiding caffeine and alcohol: Caffeine and alcohol can interfere with sleep, so it's best to avoid them in the hours leading up to bedtime.5:Engaging in regular physical activity: Regular exercise can help improve sleep quality, but it's important to avoid exercising too close to bedtime as it can interfere with sleep."
             return response, 'sleep'

    elif tag == 'stress':
             response = "It's important to make time for relaxation and stress management. Here are some activities you might find helpful:Practicing mindfulness meditation or deep breathing exercises, Taking a yoga or tai chi class that focuses on relaxation, Engaging in a hobby or activity that you enjoy, such as reading or listening to music"
             return response, 'stress'

    elif tag == 'unhealthy habits':
             response = "It's important to take small steps to change unhealthy habits over time. Here are some tips for breaking bad habits:Setting a specific goal for changing the habit, such as cutting back on smoking or drinking, Identifying triggers that lead to the habit and finding ways to avoid or cope with them, Enlisting the support of friends or family members who can help you stay on track"
             return response, 'unhealthy habits'
    

    if tag == 'not feeling well':
        
        followup = 'Please answer the following question about your symptom using 0 for absence and 1 for presence: tell me if you would like to continue with okay or no'
        return followup,tag
    if tag == "vital_signs":
       response = "Vital signs are measurements of the body's basic functions, including body temperature, blood pressure, pulse rate, and respiratory rate. Which vital sign do you want to measure?"
       return response, tag

    elif tag == "temperature":
         response = "Do you have any measurement device to record vital signs?"
         return response, tag

    elif tag == "temperature_yes" or tag == "temperature_yes_device":
        response = "Great! Let's start with the temperature. Measure the body temperature and enter the result.is that greater than 37.5 or less than 37.5"
        return response, tag

    elif tag == "vital_signs_temperature":
         temperature = float(text)
         if temperature > 37.5:
             response = "According to your response, you have a fever."
         else:
             response = "You do not have a fever."
             return response, 'vital_signs_temperature'

    elif tag == "temperature_fever":
         response = "According to your response, you have a fever."
         return response, tag
 
    elif tag == "temperature_no":
         response = "To measure your body temperature, you can use a regular thermometer that you place under your tongue. Enter the response you see on the thermometer.is that greater than 37.5 or less than 37.5"
         return response, tag

    elif tag == "temperature_no_fever":
         response = "According to your response, you do not have a fever."
         return response, tag

    elif tag == "temperature_no_device":
        response = "To measure your body temperature, you can use a regular thermometer that you place under your tongue."
        return response, tag

    elif tag == "temperature_yes_device":
         response = "Great! Let's start with the temperature. Measure the body temperature and enter the result.is it greather than 37.5 or below 37.5"
         return response, tag
    elif tag == "blood_pressure":
         response = "Do you have any measurement device to record the vital sign?"
         return response, tag

    elif tag == "blood_pressure_device_yes":
        response = "Great! Let's start with the blood pressure.Please enter the patient's systolic blood pressure and diastolic blood pressure. is it greater than or equals to 140 or 90"
        return response, tag

    elif tag == "blood_pressure_device_yes_result":
        systolic = int(text.split()[0])
        diastolic = int(text.split()[1])
        if systolic >= 140 or diastolic >= 90:
           response = "It looks like your blood pressure is above normal. We recommend you seek medical attention."
        else:
           response = "Congrats! Your blood pressure is normal."
           return response, tag

    elif tag == "blood_pressure_device_no":
         response = "If you do not have a blood pressure monitor, it may be difficult to measure your blood pressure accurately. One way is to check for symptoms that may indicate high or low blood pressure, such as feeling dizzy, lightheaded, or having a headache, which may be a sign of high blood pressure. If you feel weak or fatigued, it may be a sign of low blood pressure. Enter what you got down below to check if it's normal."
         return response, tag

    elif tag == "blood_pressure_no_device_result":
         systolic = int(text.split()[0])
         diastolic = int(text.split()[1])
         response=""
         if systolic >= 140 or diastolic >= 90:
            response = "It looks like your blood pressure is above normal. We recommend you seek medical attention."
         else:
            response = "Congrats! Your blood pressure is normal."
         return response, tag

    elif tag =="respiratory_rate":
          response = "OK, let's measure your respiratory rate. To measure your respiratory rate, you can count the number of breaths you take in 30 seconds and multiply by two.is the rate greater than 20 or less than 12"
          return response, tag
    elif tag == "respiratory_rate_result":
         rate = int(text)
         if rate > 20 or rate < 12:
            response = "Your respiratory rate is outside of the normal range. It's important to consult with a healthcare provider."
         else:
             response = "Your respiratory rate looks good."
             return response, tag
    list_of_intents= intents_json['intents']
    for i in list_of_intents:
        if tag==i['tag'] :
            result= random.choice(i['responses'])
            
            return result,tag
def response(text):
    return_list=predict_class(text,model)
    response,_=get_response(return_list,intents,text)
    return response



#@app.route('/',methods=['GET','POST'])
#@app.route('/home',methods=['GET','POST'])
#def yo():
    #return render_template('main.html')
#app = Flask(__name__)
app.secret_key = 'many random bytes'

app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'crud'

mysql = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="crud"
)
@app.route('/chat',methods=['GET','POST'])
@app.route('/home',methods=['GET','POST'])
def home():
    return render_template('index.html')

@app.route('/', methods =['GET', 'POST'])
def login():
    msg = ''

    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        cursor = mysql.cursor()
        cursor.execute("SELECT * FROM accounts WHERE username=%s AND password=%s", (username, password))
        user = cursor.fetchone()
        if user:
            session['logged_in'] = True
            session['username'] = username
            session['role'] = user[10]  
            if session['role'] == 'user':
                return redirect(url_for('Main'))
            elif session['role'] == 'admin':
                return redirect(url_for('Admin'))
        else:
            return "Invalid login credentials. Please try again."
    else:
        return render_template('login.html')
@app.route('/main')
def Main():
    if session.get('logged_in') and session['role'] == 'user':
        return render_template('main.html')
    else:
        return redirect('/')
@app.route('/logout')
def Logout():
    if session.get('logged_in') and session['role'] == 'user':
        return render_template('login.html')
    else:
        return redirect('/')

@app.route('/register', methods =['GET', 'POST'])
def register():
    msg = ''
    if request.method == 'POST': 
        if 'username' in request.form and 'password' in request.form and 'firstname' in request.form and 'lastname' in request.form and 'age' in request.form and 'gender' in request.form and 'address' in request.form and 'phoneno' in request.form and 'email' in request.form and 'role' in request.form:
            username = request.form['username']
            firstname = request.form['firstname']
            lastname = request.form['lastname']
            password = request.form['password']
            age = request.form['age']
            gender = request.form['gender']
            phoneno = request.form['phoneno']
            address =  request.form['address']
            email = request.form['email']
            role = request.form['role']
            cursor = mysql.cursor()
            cursor.execute('INSERT INTO accounts (username, password, email, gender, firstname, lastname, age, phoneno, address, role) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)', (username,firstname, lastname, age, gender, phoneno, address, password, email, role))
            mysql.commit()
            msg = 'You have successfully registered !'
            return redirect(url_for('login'))
    elif request.method == 'POST':
        msg = 'Please fill out the form !'
    return render_template('register.html', msg = msg)

@app.route('/admin')
def Admin():
    cursor = mysql.cursor()
    cursor.execute("SELECT * FROM facility")
    data = cursor.fetchall()
    cursor.close()

    return render_template('admin.html', facility=data)

@app.route('/insert', methods = ['POST'])
def insert():
    if request.method == "POST": 
        flash("Data Inserted Successfully")
        hname = request.form['hname']
        hospital_type = request.form['hospital_type']
        rating = request.form['rating']
        working_hour = request.form['working_hour']
        severity_levels=request.form['severity_levels']
        cur = mysql.cursor()
        cur.execute("INSERT INTO facility (hname, hospital_type, rating, working_hour,severity_levels) VALUES (%s, %s, %s, %s,%s)", (hname, hospital_type, rating, working_hour, severity_levels))
        mysql.commit()
        return redirect(url_for('Admin'))

@app.route('/delete/<string:id_data>', methods = ['GET'])
def delete(id_data):
    flash("Record Has Been Deleted Successfully")
    cur = mysql.cursor()
    cur.execute("DELETE FROM facility WHERE id=%s", (id_data,))
    mysql.commit()
    return redirect(url_for('Admin'))

@app.route('/update', methods= ['POST', 'GET'])
def update():
    if request.method == 'POST':
        
        flash("Data Inserted Successfully")
        hname = request.form['hname']
        hospital_type = request.form['hospital_type']
        rating = request.form['rating']
        working_hour = request.form['working_hour']
        severity_levels = request.form['severity_levels']

        cur = mysql.cursor()
        cur.execute("""
        UPDATE facility SET hname=%s, hospital_type=%s, rating=%s, working_hours=%s, severity_levels=%S
        WHERE id=%s
        """, (hname, hospital_type, rating, working_hour, severity_levels))
        flash("Data Updated Successfully")
        return redirect(url_for('Admin'))
@app.route('/hospital_recommendation', methods=['POST'])
def hospital_recommendation():
    # Get the user's input
    user_input = request.form['text']

    # Connect to the database
    cur = mysql.connection.cursor()

    # Check if the user prefers private or public hospitals
    if 'private' in user_input.lower():
        hospital_type = 'private'
        cur.execute("SELECT hname, rating, working_hour  FROM facility WHERE hospital_type='private' ORDER BY rating DESC", ('%breast cancer%', hospital_type))
        results = cur.fetchall()
        if len(results) == 0:
            return 'Sorry, we could not find any private hospitals for breast cancer treatment.'
        else:
            response = 'Here are some private hospitals that we recommend for breast cancer treatment: '
            for result in results:
                response += f'{result[0]} - {result[1]}'
            return response
    elif 'public' in user_input.lower():
        hospital_type = 'public'
        cur.execute("SELECT hname, rating, working_hour  FROM facility WHERE hospital_type='public'  ORDER BY rating DESC", ('%breast cancer%', hospital_type))
        results = cur.fetchall()
        if len(results) == 0:
            return 'Sorry, we could not find any public hospitals for breast cancer treatment.'
        else:
            response = 'Here are some public hospitals that we recommend for breast cancer treatment:\n'
            for result in results:
                response += f'{result[0]}\n'
            return response
            cur.close()
    else:
        return 'Sorry, I didn\'t understand your preference. Please try again.'




@app.route("/get")
def chatbot():
    userText = request.args.get('msg')
    resp = response(userText)
    return resp








# route for chatbot response





