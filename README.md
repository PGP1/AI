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



 NEWEST MODEL is a saved_model object which is the most current model trained this model may change over time. 