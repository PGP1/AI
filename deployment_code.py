
import sagemaker
from sagemaker import get_execution_role
role = get_execution_role()


# import tarfile

# with tarfile.open('model.tar.gz', mode='w:gz') as archive:
#     archive.add('1', recursive=True)

# ^^^ if your model is not a tar file yet and is not a 
    


sagemaker_session = sagemaker.Session()
inputs = sagemaker_session.upload_data(path='model.tar.gz', key_prefix='model')

from sagemaker.tensorflow.serving import Model

model = Model(  model_data = 's3://' + sagemaker_session.default_bucket() + '/model/model.tar.gz',  #s3 bucket location 
                role=role,
                entry_point = 'model_training_script.py', #this is the script which builds your model.
                framework_version='2.1.0'
             )

predictor = model.deploy(initial_instance_count=1, instance_type='ml.t2.medium') #provides you a predictor object locally which you can 
                                                                                 # use to make trial predictions

event = {"signature_name":"predict", "inputs":{"water_level": 7, "temperature_level": 38, "ldr": 400, "pH": 7, "humidity":70}}

predictor.predict(event)