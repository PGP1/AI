import tensorflow as tf

tf.compat.v1.disable_eager_execution()


model_url = "newest_model"

imported = tf.keras.models.load_model(model_url)

def predict_fn(x):
    example = tf.train.Example()
    example.features.feature['water_level'].float_list.value.extend([x['water_level']])
    example.features.feature['temperature_level'].float_list.value.extend([x['temperature_level']])
    example.features.feature['ldr'].float_list.value.extend([x['ldr']])
    example.features.feature['pH'].float_list.value.extend([x['pH']])
    example.features.feature['humidity'].float_list.value.extend([x['humidity']])
    return imported.signatures["predict"](examples=tf.constant([example.SerializeToString()]))


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

print(len(class_names))

id_name=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25
            ,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48
            ,49,50,51,52,53]


def input_fn(features, batch_size=256):
    return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)

features = ['water_level', 'temperature_level', 'ldr', 'pH','humidity']
predict = {}

print("Please type numeric values as prompted.")
print(features)

for feature in features:
    val = input(feature + ": ")
    predict[feature] = float(val)

print(predict)    
predictions = predict_fn(predict)

print(predictions)
class_id = predictions['class_ids'][0]
probs = predictions['probabilities']
init = tf.compat.v1.global_variables_initializer()

print(class_id)

with tf.compat.v1.Session() as sess:
    
    sess.run(init)
    print(sess.run(class_id).item())
    c_id = sess.run(class_id).item()
    
    print(sess.run(probs)[0][c_id])
    
    probability = sess.run(probs)[0][c_id]
    
    print(predictions['class_ids'])
    print('Prediction is "{}" ({:.1f}%)'.format(class_names[c_id], 100 * probability))

