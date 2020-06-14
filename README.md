# Tensorflow Model for Plantly
Description: contains the code which trains the model based on the test_d and updated_data sets. 

> NOTE: in order for the model to save you must create a folder called tensormodels within the same directory as the this script. Models will be exported to that directory. 

<h1> 1. Pre-Requistites </h1>

Dependencies
   - tensorflow
   - numpy
   - pandas 
  

 RSPI test script : 
Dependencies

Tensorflow 
   ``` bash
       pip install tensorflow
   ```

NOTE: this script can be run locally as long as you have tensorflow and a saved_model type object in the same directory as this script. The script takes in values, for water the value must be with in 1-12 , ph: 4-8, humidity:1-100, light: 200-800, temp:5-40 in order to get accurate result. 

 NEWEST_MODEL is a saved_model object which is the most current model trained this model may change over time. 

 predict_function_code :
  - this python file contains code which can be imported so that you can obtain a prediction
  - parameters : a dictionary of values {}
    - returns a string representing the prediction.

We are no longer using a self-managed cluster so the only pre-requisite is an AWS account to create an AWS managed Elasticsearch Cluster

Note: be sure to have your SNS queue and SNS Topic set up and working before integrating the deployment of the model.


<h1> 2. Setup and Training </h1>


1) A good platform to use for machine learning based code and scripts is Jupyter Notebooks so we are going to have to set that up either locally with ANACONDA 3 or SageMaker which requires an instance. LOCALLY: install anaconda 3 from the internet and choose your desired language and libraries. If you want to use the skeleton, you can select the tensorflow 2.1.0 with python 2.7.

 
2) Once you have your ipynb notebook you can write your own machine learning code or use the skeletons provided in this repo.
The code in this repo, builds and trains a estimator or more specifically a DNNClassifier which helps us identify plant conditions.


3) Upload the data into skeleton by providing a data csv within the local instance located in the same directory as the ipynb file.
In the local directory on the side nav of the interface you should be able to see your files there, be sure to have 2 sets of .csv data: one for tests and one for training. (if using skeleton, training data should be named 'updated.csv' and 'test_d.csv')


4) Once the data is in train the model by running the code within the notebook itself, once done the model should be exported to the local directory automatcally, under the name as be deployed to the s3 bucket as well.
![s3 bucket](/images/s3)
<h1>3. Deployment </h1>

1) Login into AWS and engage the SageMaker service.
   
2) Set up your SageMaker Instance and JupyterLabs. (note: ml.t2.medium is the cheapest instance size so keep in mind how big your model is before your deploy.)

3) Upload local files to the notebook. This includes the training script which is used to make the model as this will be required in the deployment code for the Sagemaker endpoint. 

4) Run the deployment code inside of jupyterlabs and wait for your endpoint to finish creation

5) Once deployment is finished, the endpoint can be used and takes in body data from the format : 
```
{"signature_name":"predict", "inputs":{"water_level": 7, "temperature_level": 38, "ldr": 400, "pH": 7, "humidity":70}}
```
this will return a dictionary object, with the class id of categorized by the model. this value can be passed to the array inside of 
the predict_function code. 


<h1> 4. Testing </h1>

In order to test and debug, 
- Set up cloudwatch inside of the amazon cloud services by enabling the service
- Create a test configuration inside of your lambda function which triggers your lambda based off of a json dummy which we can put in
