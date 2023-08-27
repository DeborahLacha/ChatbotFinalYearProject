from chatbot import app
from flask import Flask, render_template, request, url_for, flash, session, jsonify
# from chatbot_codes.full_code import recommend_hospital
from chatbot.forms import chatbotform
from chatbot.__init__ import model, words, classes, intents
from werkzeug.utils import redirect
from flask_mysqldb import MySQL
import mysql.connector
import re
import pymysql

import nltk
import pickle
import json
import numpy as np
from keras.models import Sequential, load_model
import random
from datetime import datetime
import pytz
import requests
import os

import pandas as pd
import time


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

   # if tag == 'hospital_recommendation':
      #  return 'Sure, I can recommend some hospitals based on your preferences. Do you prefer private or public hospitals?', tag

    if tag == 'hospital_recommendation':
            response = "Sure, I can recommend some hospitals based on your preferences. Do you prefer private or public hospitals?"
            response, tag

    if tag == 'private_hospital_recommendation':
        # Fetch hospitals from database based on rating
        cur = mysql.cursor()

        cur.execute(
            "SELECT hname, rating FROM facility WHERE hospital_type='private' ORDER BY rating DESC", ('%breast cancer%',))
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
            "SELECT hname, rating FROM facility WHERE hospital_type='public' ORDER BY rating DESC", ('%breast cancer%',))
        results = cur.fetchall()
        if len(results) == 0:
            response = 'Sorry, we could not find any public hospitals for breast cancer treatment.'
        else:
            response = 'Here are some public hospitals that we recommend for breast cancer treatment:\n'
            for result in results:
                response += f'{result[0]} - {result[1]}\n'
        return response, tag


    if tag == 'not feeling well':
            # Get the symptom inputs from the form
            data = pd.read_csv(
                "C:/Users/HU/Desktop/Chatbot/chatbot_codes/breast_cancer_symptoms.csv")

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

            # Evaluate the model on the testing set
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            print("Accuracy:", accuracy)

            # Ask the user for each symptom and predict the severity level
            print("Please answer the following questions about your symptoms:")
            severity_mapping = {"Low severity": 1,
                "Medium severity": 2, "High severity": 3}
            symptoms = []
            for feature in X.columns:
                value = input(f"{feature}: ")
                symptoms.append(int(value))
            symptom_data = pd.DataFrame([symptoms], columns=X.columns)
            severity_levels = model.predict(symptom_data)
            severity_levels = [severity_mapping[level]
                for level in severity_levels]
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
            # Render the severity classification template with the results

        # If the request method is GET, render the severity classification form
            response = f"Based on your symptoms, your severity level is {severity_category}. {medical_attention}"
            
    #return render_template('result.html', severity_category=severity_category, medical_attention=medical_attention)

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
        cur = mysql.cursor()
        cur.execute("INSERT INTO facility (hname, hospital_type, rating, working_hour) VALUES (%s, %s, %s, %s)", (hname, hospital_type, rating, working_hour))
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

        cur = mysql.cursor()
        cur.execute("""
        UPDATE facility SET hname=%s, hospital_type=%s, rating=%s, working_hours=%s
        WHERE id=%s
        """, (hname, hospital_type, rating, working_hour))
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
        cur.execute("SELECT hname, rating FROM facility WHERE hospital_type='private' ORDER BY rating DESC", ('%breast cancer%', hospital_type))
        results = cur.fetchall()
        if len(results) == 0:
            return 'Sorry, we could not find any private hospitals for breast cancer treatment.'
        else:
            response = 'Here are some private hospitals that we recommend for breast cancer treatment:\n'
            for result in results:
                response += f'{result[0]} - {result[1]}\n'
            return response
    elif 'public' in user_input.lower():
        hospital_type = 'public'
        cur.execute("SELECT hname FROM facility WHERE hospital_type='public'  ORDER BY rating DESC", ('%breast cancer%', hospital_type))
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





