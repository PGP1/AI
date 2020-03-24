import numpy as np 
import pandas as pd 

import tensorflow as tf
import tensorflow.contrib.eager as tfe 
import os 


tf.enable_eager_execution()

train_dataset_url = ""

train_dataset_fp = tf.keras.utils.get_file(fname=os.path.basename(train_dataset_url), origin=train_dataset_url)

column_names = ['water_level','pH','temperature','LDR','labels']
feature_names = column_names[:-1]
label_name = column_names[-1]

batch_size = 32

train_dataset = tf.contrib.data.make_csv_dataset(

		train_dataset_fp,
		batch_size,
		column_names=column_names,
		label_name=label_name,
		num_epochs=1)


features, labels = next(iter(train_dataset))

print(features)





