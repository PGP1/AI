#!/usr/bin/env python3

import socket
import boto3
#package for sending obj as bytes
import pickle
import atexit
import json
import math
#import modelPredict
from decimal import * 
import urllib.parse
import boto3
import re
from predict_function_code import model_predict



s3 = boto3.client('s3')

queue_url = 'https://sqs.ap-southeast-2.amazonaws.com/624634462175/s3_updated'

sqs_client = boto3.client('sqs')

def iterate_bucket_items(bucket, prefix, ignore):
    """
    Generator that iterates over all objects in a given s3 bucket

    See http://boto3.readthedocs.io/en/latest/reference/services/s3.html#S3.Client.list_objects_v2 
    for return data format
    :param bucket: name of s3 bucket
    :return: dict of metadata for an object
    """
    pattern = re.compile("{}\/((?!{}).*)\/.*\.json".format(prefix, ignore))


    paginator = s3.get_paginator('list_objects_v2')
    operation_parameters = {'Bucket': bucket,
                            'Prefix': prefix}
                            
    types = ['ldr', 'water', 'ph', 'temp', 'humidity']
    types.remove(ignore)
    
    page_iterator = paginator.paginate(**operation_parameters)

    get_last_modified = lambda obj: int(obj['LastModified'].strftime('%s'))

    for page in page_iterator:
        if page['KeyCount'] > 0:
            print("latest", sorted(page['Contents'], key=get_last_modified))
            for item in sorted(page['Contents'], key=get_last_modified):
                key = item['Key']
                if(pattern.match(key)):
                    type = key.split("/")[1]
                    if(type in types):
                        types.remove(type)
                        yield item


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
    
    
def get_messages_from_queue(url):
    """Generates messages from an SQS queue.

    This continues to generate messages until the queue is empty.
    Every message on the queue will be deleted.

    :param queue_url: URL of the SQS queue to drain.

    """

    while True:
        resp = sqs_client.receive_message(
            QueueUrl=queue_url,
            AttributeNames=['All'],
            MaxNumberOfMessages=10
        )

        try:
            yield from resp['Messages']
        except KeyError:
            return

        entries = [
            {'Id': msg['MessageId'], 'ReceiptHandle': msg['ReceiptHandle']}
            for msg in resp['Messages']
        ]

        resp = sqs_client.delete_message_batch(
            QueueUrl=queue_url, Entries=entries
        )

        if len(resp['Successful']) != len(entries):
            raise RuntimeError(
                f"Failed to delete messages: entries={entries!r} resp={resp!r}"
            )
        else:
            break
            
    
if __name__ == '__main__':
    
   # 1 grab events from sqs queue
    event = []
        
    for msg in get_messages_from_queue(queue_url):
            #print(msg)
            event.append(msg)
            

    #print(event)
    
    events = json.loads(event[0]["Body"])

    message= json.loads(events["Message"])

    key = message["Records"][0]["s3"]["object"]["key"]
    
    bucket_name = message['Records'][0]['s3']['bucket']['name']
    
    print("key:", key)
    print("bucket_name:", bucket_name)

    data = key.split("/")
    print("data:", data)
    
    folder = data[0]
    print("foldeR:", folder)
    
    currentMeasure = data[1]
    print("currentMeasure : " , currentMeasure)
    
    try:
        #3 - Fetch the file from S3
        response = s3.get_object(Bucket=bucket_name, Key=key)
        print("response", response)
        #4 - Deserialize the file's content
        text = response["Body"].read().decode()
        data = json.loads(text)
        
    
        features = ['water_level', 'temperature_level', 'ldr', 'pH','humidity']
        feat_data = {}
        feat_data[currentMeasure] = data['value']
    
        # response = s3.get_object(Bucket=bucket, Key='ghzy567-test7/LDR')

        # #4 - Deserialize the file's content
        # text = response["Body"].read().decode()
        # data = json.loads(text)

        
        for i in iterate_bucket_items(bucket_name, folder, ignore=currentMeasure):
            print("i", i)
            page_key = i["Key"].split("/")[1]
            temp = s3.get_object(Bucket=bucket_name, Key=i["Key"])
            bod = temp["Body"].read().decode()
            da = json.loads(bod)
            feat_data[page_key] = da["value"]
            
    #5 - Print the content
        print("feat_data", feat_data)
       
        feat_data['water_level'] = formatWater(feat_data['water'])
        feat_data['temperature_level'] = feat_data['temp']
        feat_data['pH'] = feat_data['ph']
        
        del feat_data['ph']
        del feat_data['temp']
        del feat_data['water']
        
        print("formatted feat_data", feat_data)
        #6 - Parse and print the transactions
        
        prediction = model_predict.predict(feat_data)

    except Exception as e:
        print(e)
        raise e

    
    

    
    
    
    
    """
    message = json.loads(event["Records"][0]["Sns"]["Message"])
    
    print(message)
    
     #2 - Get the file/key name
    key = urllib.parse.unquote_plus(message['Records'][0]['s3']['object']['key'], encoding='utf-8')
    print(bucket)
    data = key.split("/")
    folder = data[0]
    currentMeasure = data[1]
    print("currentMeasure : " , currentMeasure)
    """

