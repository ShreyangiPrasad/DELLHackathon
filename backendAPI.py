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
	df = pandas.read_csv(r'.\X_validation.csv')
		# csvreader = csv.reader(csvfile)
  	# for row in :
   	# 	rows.append(row)
	fileName=r'.\xgboost_model.pkl'
	model_xgboost = pickle.load(open(fileName, 'rb'))

	fileName1 = r'.\svm_model.pkl'
	model_svm = pickle.load(open(fileName1, 'rb'))
	
	text_data = df.iloc[9]
	# text_data = clean_text(text_data)
	# subject_data = request_data['subject']
	# subject_data = clean_text(subject_data)

	svm_prediction = model_svm.predict([text_data])
	
	if(svm_prediction==1):
		xgboost_prediction = model_xgboost.predict([text_data])[0]
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