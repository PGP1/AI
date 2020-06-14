import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import os 

test_path = 'https://sagemaker-ap-southeast-2-624634462175.s3-ap-southeast-2.amazonaws.com/test_d.csv'

train_dataset_fp = "https://sagemaker-ap-southeast-2-624634462175.s3-ap-southeast-2.amazonaws.com/updated_data.csv"

print("local copy of the dataset file: {}".format(train_dataset_fp))
  
#inserting datset and seperating data into data frames and tensors.
column_names = ['water_level','temperature_level','ldr','pH','humidity','label']
feature_names = column_names[:-1]
label_name = column_names[-1]

train = pd.read_csv(train_dataset_fp, names=column_names, header=0)
test = pd.read_csv(test_path, names=column_names, header=0)

train_y = train.pop('label')
test_y = test.pop('label')

def input_fn(features, labels, training=True, batch_size=256):
        dataset = tf.data.Dataset.from_tensor_slices((dict(features),labels))    
        if training:
            dataset= dataset.shuffle(1000).repeat()
            
        return dataset.batch(batch_size)

def get_np(traindf, features):
    
    def _z_score_params(column):
        mean = traindf[column].mean()
        std = traindf[column].std()
        return {'mean': mean, 'std': std}

    normalization_parameters = {}
    for column in features:
        normalization_parameters[column] = _z_score_params(column)
    return normalization_parameters
    
normalization_param = get_np(train,feature_names) 

    
def _make_zscaler(mean, std):
    def zscaler(col):
        return (tf.cast(col,dtype=tf.float32) - mean)/std
    return zscaler

def make_tf_normalised_column():
        normalised_f = []
        for column_name in feature_names:
            column_params = normalization_param[column_name]
            mean = column_params['mean']
            std = column_params['std']
            normaliser_fn = _make_zscaler(mean,std)
            normalised_f.append(tf.feature_column.numeric_column(column_name,normalizer_fn=normaliser_fn))
        return normalised_f
# the 4 methods above normalize the dataset to make it easier for the model to process and build. (makes the model more 
# accurate as well)    
my_feature_columns = []

my_feature_columns= make_tf_normalised_column()

print(my_feature_columns[:1])

classifier = tf.estimator.DNNClassifier(
     feature_columns=my_feature_columns,       
     hidden_units=[100,54],
     n_classes = 54,
     optimizer=lambda: tf.compat.v1.train.AdamOptimizer(
        learning_rate=tf.compat.v1.train.exponential_decay(
            learning_rate=0.03,
            global_step=tf.compat.v1.train.get_global_step(),
            decay_steps=10000,
            decay_rate=0.96))

)

#the official building of the model along with its layers.
#training of the model. 
classifier.train(
    input_fn=lambda: input_fn(train,train_y, training=True),
    steps = 150000
)

eval_result = classifier.evaluate(input_fn=lambda: input_fn(test, test_y, training=False))
print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))
# evaluates the models accuracy with some test data.


feature_specs = {
    "water_level": tf.compat.v1.keras.backend.placeholder(shape=(None, ), dtype=tf.float32, name='input_water'),
    "temperature_level":  tf.compat.v1.keras.backend.placeholder(shape=(None, ), dtype=tf.float32, name='input_temp'),
    "ldr":  tf.compat.v1.keras.backend.placeholder(shape=(None, ), dtype=tf.float32, name='input_ldr'),
    "pH":  tf.compat.v1.keras.backend.placeholder(shape=(None, ), dtype=tf.float32, name='input_pH'),
    "humidity": tf.compat.v1.keras.backend.placeholder(shape=(None, ), dtype=tf.float32, name='input_hum')  
}

serving_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(feature_specs)

# def input_r_fn():
#     serialized_tf_example = tf.compat.v1.placeholder(dtype=tf.string, shape=[1], 
#                                            name='data')
    
#     receiver_tensors = {'examples': serialized_tf_example}
#     features = tf.compat.v1.parse_example(serialized_tf_example,  feature_specs)
#     return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)


# my_f = tf.feature_column.make_parse_example_spec(my_feature_columns)

# print(my_f)
# input_receiver = tf.estimator.export.build_raw_serving_input_receiver_fn(
#     my_f, default_batch_size=None
# )

classifier.export_saved_model(export_dir_base='/',
                              serving_input_receiver_fn=serving_fn
                              )

#set up of the reciever tensors and the feature spec so that the model can be used as a Save_model object
#once exported.


