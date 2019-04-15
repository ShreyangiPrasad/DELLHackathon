from flask import Flask, request, jsonify
import pickle
import re
from bs4 import BeautifulSoup
import json
import requests
import pandas as pd
import base64
import csv

app = Flask(__name__)

def application(environ, start_response):
  if environ['REQUEST_METHOD'] == 'OPTIONS':
    start_response(
      '200 OK',
      [
        ('Content-Type', 'application/json'),
        ('Access-Control-Allow-Origin', '*'),
        ('Access-Control-Allow-Headers', 'Authorization, Content-Type'),
        ('Access-Control-Allow-Methods', 'POST'),
      ]
    )
    return ''

def classify(request_data):

	rows = []
	filename = "C:\Users\shreyangi_prasad\Desktop\hackathon\X_validation.csv"
	with open(filename, 'r') as csvfile:
		csvreader = csv.reader(csvfile)
    	for row in csvreader: 
        	rows.append(row) 


	svm_clf = pickle.load(open("./svm_model.pkl","rb"))
	xgboost_clf = pickle.load(open("./xgboost.pkl","rb"))
	
	text_data = rows[0]
	# text_data = clean_text(text_data)
	# subject_data = request_data['subject']
	# subject_data = clean_text(subject_data)

	svm_prediction = svm_clf.predict([text_data])
	
	if(svm_prediction==1):
		xgboost_prediction = xgboost_clf.predict([text_data])[0]
		failure=xgboost_prediction
	else:
		failure=0

	response_obj = {}
	response_obj['failure'] = xgboost_prediction

@app.route('/classifyPost', methods=['GET', 'POST'])
def classifyPost():
	if(request.data):
		request_data = request.json
		if('text' in request_data.keys() and 'subject' in request_data.keys()):
			response_object = classify(request_data)
		else:
			response_object = {'error': 'Bad Request'}
	else:
		response_object = {'error': 'Bad Request'}

	json_response = app.response_class(response=json.dumps(response_object), status=200, mimetype='application/json')
	return(json_response)

if __name__ == '__main__':
	app.run()