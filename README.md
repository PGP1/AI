Elasticsearch
Tests, configs and setup instructions for our Elasticsearch Cluster

1 Pre-Requistites

TRAINING SCRIPT : 
 - what to install:
   tensorflow
   numpy
   pandas 
  
 - contains the code which trains the model based on the test_d and updated_data sets. NOTE: in order for the model to save you must create a folder called tensormodels within the same directory as the this script. Models will be exported to that directory. 

 RSPI test script : 
 - what to install:
   tensorflow (pip install tensorflow ..)

 - this script can be run locally as long as you have tensorflow and a saved_model type object in the same directory as this script. The script takes in values, for water the value must be with in 1-12 , ph: 4-8, humidity:1-100, light: 200-800, temp:5-40 in order to get accurate result. 

 NEWEST_MODEL is a saved_model object which is the most current model trained this model may change over time. 

 predict_function_code :
  - this python file contains code which can be imported so that you can obtain a prediction
  - parameters : a dictionary of values {}
    - returns a string representing the prediction.

We are no longer using a self-managed cluster so the only pre-requisite is an AWS account to create an AWS managed Elasticsearch Cluster

Note: be sure to have your SNS queue and SNS Topic set up and working before integrating the deployment of the model.


2 Code Setup 

2.1 a good platform to use for machine learning based code and scripts is Jupyter Notebooks so we are going to have to set that up either locally with ANACONDA 3 or SageMaker which requires an instance. LOCALLY: install anaconda 3 from the internet and choose your desired language and libraries. If you want to use the skeleton, you can select the tensorflow 2.1.0 with python 2.7.

2.2 Once you have your ipynb notebook you can write your own machine learning code or use the skeletons provided in this repo.
The code in this repo, builds and trains a estimator or more specifically a DNNClassifier which helps us identify plant conditions.

2.3 Upload the data into skeleton by providing a data csv within the local instance located in the same directory as the ipynb file.
In the local directory on the side nav of the interface you should be able to see your files there, be sure to have 2 sets of .csv data: one for tests and one for training. (if using skeleton, training data should be named 'updated.csv' and 'test_d.csv')

2.4 Once the data is in train the model by running the code within the notebook itself, once done the model should be exported to the local directory automatcally, under the name as be deployed to the s3 bucket as well.

3 Deployment

3.1 Login into AWS and engage the SageMaker service.
   
3.2 Set up your SageMaker Instance and JupyterLabs. (note: ml.t2.medium is the cheapest instance size so keep in mind how big your model is before your deploy.)

3.3 Upload local files to the notebook. This includes the training script which is used to make the model as this will be required in the deployment code for the Sagemaker endpoint. 

3.4 Run the deployment code inside of jupyterlabs and wait for your endpoint to finish creation


4 Testing

4.1 In order to test and debug, set up cloudwatch inside of the amazon cloud services by enabling the service
and create a test configuration inside of your lambda function which triggers your lambda based off of a json dummy which we can put in
