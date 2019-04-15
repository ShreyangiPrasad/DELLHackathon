from flask import Flask, request, jsonify
import pickle
import re
from bs4 import BeautifulSoup
import json
import requests
import pandas as pd
import base64
import csv
import pandas

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
	#rows = []
	#filename = r'C:\\Users\shreyangi_prasad\\Desktop\\hackathon\\X_validation.csv'
	df = pandas.read_csv('.\X_validation.csv')
		# csvreader = csv.reader(csvfile)
  	# for row in :
   	# 	rows.append(row)


	with open('.\model_Xgboost', 'rb') as f:
		model_xgboost = pickle.load(f)
		
	svm_clf = pickle.load(open(r'svm_model',"rb"))
	xgboost_clf = pickle.load(open(r'model_Xgboost',"rb"))
	
	text_data = df[0]
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
	response_obj['failure'] = failure
	return response_obj

@app.route('/')
def helloworld():
	return "hello world"

@app.route('/classifyPost', methods=['POST'])
def classifyPost():
	if(request.data):
		request_data = request.json
		if('service_tag' in request_data.keys()):
			response_object = classify(request_data)
		else:
			response_object = {'error': 'Bad Request'}
	else:
		response_object = {'error': 'Bad Request'}

	json_response = app.response_class(response=json.dumps(response_object), status=200, mimetype='application/json')
	return(json_response)

if __name__ == '__main__':
	app.run(debug=True)