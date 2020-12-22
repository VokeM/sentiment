# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 10:40:13 2020

@author: Voilet Pince
"""

import pandas as pd
import flask
from flask import Flask
import joblib
import numpy as np
from flask_cors import CORS
import warnings
import mysql.connector as connection

try: 
    mydb = connection.connect(host='localhost', database = 'review',user='root', passwd='',use_pure=True)#creating the connection to the database review   
    query = "Select * from review;"#selecting the table review from the database review    
    ds= pd.read_sql(query,mydb)#storing the review table in a Pandas dataframe
    mydb.close() #close the connection
#exception to catch errors and print the errors
except Exception as e:
        mydb.close()
        print(str(e))

warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)
@app.route('/sentiment/', methods=['GET'])

def make_prediction():
    if flask.request.method == 'GET':
        X2 = ds['Comment'].values
        X_new = cv.transform(X2)# building up feature vector of our input
        prediction = clf.predict(X_new) #making predictions
        zeros = int(np.sum(prediction == 0))#counting the number of zeros in the analysis
        ones = int(np.sum(prediction == 1))#counting the number of ones in the analysis
        return flask.jsonify(zeros,ones)#returning the zeros and ones
    
if __name__ == '__main__':
    clf = joblib.load("NBmodel.pkl")#loading the already trained model
    cv = joblib.load("vector.pkl")#loading the count vectorizer that will be used to vectorize the new inputs
    app.run(debug=True, use_reloader=False,threaded=False)
