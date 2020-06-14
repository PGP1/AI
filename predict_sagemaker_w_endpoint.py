import os
import io
import boto3
import urllib.parse
import json
import csv
import math
import re
import datetime
import requests
import random
from requests_aws4auth import AWS4Auth

# grab environment variables
runtime= boto3.client('runtime.sagemaker')
ENDPOINT_NAME = os.environ['model_endpoint']
s3_bucket = boto3.client('s3')

credentials = boto3.Session().get_credentials()
print("access_key", credentials.access_key, 
     "secret_key", credentials.secret_key, 
     "id_token", credentials.token)
     
     

# set credentials 
region = 'ap-southeast-2' # e.g. us-west-1
service = 'es'
credentials = boto3.Session().get_credentials()
awsauth = AWS4Auth(credentials.access_key, credentials.secret_key, region, service, session_token=credentials.token)

# set elasticsearch cluster, the Amazon ES domain, including https://
host = 'https://search-plantly-es-cheap-my4i72dmshwihajjj2sbwqii3i.ap-southeast-2.es.amazonaws.com' 

# headers for http equest
headers = { "Content-Type": "application/x-ndjson" }

# query to test if cluster is returning data
_id =  'c0cb03a49a754a17b07b85c4d4f19039-test'
url = host + '/' + _id + '/_msearch'

query_body = [
    {},
    {"query": {"term": {"type": "water"}},"size": 1,"sort": [{"time": { "order": "asc" }}]},
    {},
    {"query": {"term": {"type": "humidity"}},"size": 1,"sort": [{"time": { "order": "asc" }}]},
    {},
    {"query": {"term": {"type": "ph"}},"size": 1,"sort": [{"time": { "order": "asc" }}]},
    {},
    {"query": {"term": {"type": "temp"}},"size": 1,"sort": [{"time": { "order": "asc" }}]},
    {},
    {"query": {"term": {"type": "ldr"}},"size": 1,"sort": [{"time": { "order": "asc" }}]},
]

data_to_post = '\n'.join(json.dumps(d) for d in query_body) + "\n"

def formatWater(waterVal):
    fval = float(float(waterVal)/100)*2
    fDec = str(float(fval)).split('.')[1]
    waterScaled = 0
    if float(fDec) >= 5:
        waterScaled = math.ceil(fval)
    if float(fDec) < 5 and float(fDec) > 0:
        waterScaled = math.floor(fval)
    elif float(fDec) == 0:
        waterScaled = fval
    return waterScaled

def lambda_handler(event, context):
    # sent and print if succesfful
    event = json.loads(json.dumps(event['body']))
    user = json.loads(event)

    _id = '{}-{}'.format(user['device'], user['username'])
    url = host + '/' + _id + '/_msearch'
    # loadin in the event and then formatting data so that can be processed.
    elastic_r = requests.get(url, auth=awsauth, data=data_to_post, headers=headers)
    raw_data = json.loads(elastic_r.text)

    latest_data = []

    for r in raw_data['responses']:
        latest_data.append(r["hits"]["hits"][0]["_source"]["value"])

    feat_data = {}
    feat_data['water_level'] = formatWater(latest_data[0])
    feat_data['humidity'] = latest_data[1]
    feat_data['pH'] = latest_data[2]
    feat_data['temperature_level'] = latest_data[3]
    feat_data['ldr'] = latest_data[4]
     # Specifying the feature data so that the data can be used for prediction.
    prediction_input = {'signature_name':"predict", 'inputs': feat_data }

    class_names = ['normal','low light','high light', 'low temp','low temp & low light',
                        'low temp & high light','high temp','high temp & low light',
                        'high temp & high light','low water','low water & low light',
                        'low water & high light', 'low water & low temp',
                        'low water & low temp & low light', ' low water & low temp & high light',
                        'low water & high temp','low water & high temp & low light',
                        'low water & high temp & high light','low ph','high ph','low light & low ph'
                        'low light & high ph','low temp & low ph','high light & low ph','high light & high ph',
                        'low temp & low ph','low temp & high ph', 
                        'low temp & low light & low ph', 'low temp & low light & high ph',
                        'low temp & high light & low ph', 'low temp & high light & high ph',
                        'high temp & low ph', 'high temp & high ph', 'high temp & low light & low ph',
                        'high temp & low light & high ph','high temp & high light & low ph',
                        'high temp & high light & high ph', 'low water & low ph', 'low water & high ph',
                        'low water & low light & low ph', 'low water & low light & high ph', 
                        'low water & high light & low ph', 'low water & high light & high ph',
                        'low water & low temp & low ph', 'low water & low temp & high ph',
                        'all levels are low','low water & low temp & low light & high ph',
                        'low water & low temp & high light & low ph',
                        'low water & low temp & high light & high ph',
                        'low water & high temp & low ph','low water & high temp & high ph',
                        'low water & high temp & low light & low ph' ,
                        'low water & high temp & low light & hight ph' ,
                        'low water & high temp & high light & low ph' ,
                        'low water & high temp & high light & high ph' ]
    # classnames for us to identify the prediction itself. 
    response = runtime.invoke_endpoint(EndpointName=ENDPOINT_NAME,
                                      ContentType='application/json',
                                      Body=json.dumps(prediction_input))
    result = json.loads(response['Body'].read().decode())
    # endpoint invocation and result decoding.
    class_index = result['outputs']['class_ids'][0][0]
    
    prediction_value = class_names[class_index]
    
    print("THIS IS MY PREDICTION", prediction_value)
    
    write_json = {}
    
    write_json['pi-id'] = user['device']
    write_json['prediction'] = prediction_value
    write_json['type'] = "prediction"
    write_json['time'] = datetime.datetime.now().isoformat()
    # formatting the result so that the data can be written to the bucket for s3.
    
    key = _id + '/' + 'predictions' + '/' + str(random.randint(0, 999999)) + '-prediction.json'
    
    s3_bucket.put_object(Body=json.dumps(write_json), Bucket='iot-plant-data', Key=key)
    
    return { 'statusCode': 200,
        'body': json.dumps(write_json)
    }
