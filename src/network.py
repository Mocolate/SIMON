import numpy as np
import tensorflow as tf

data_path = 'wsdata.tfrecords'

def read(filename_queue):
    feature = {'image_raw': tf.FixedLenFeature([], tf.string),
               'meta_code': tf.FixedLenFeature([], tf.int64)}
   
    # Define a reader and read the next record
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
   
    # Decode the record read by the reader
    features = tf.parse_single_example(serialized_example, features=feature)
   
    # Decode the image from jpeg encoding
    image = tf.image.decode_jpeg(features['image_raw'])
    
    # Cast label data into int32
    label = tf.cast(features['meta_code'], tf.int32)

    return image, label

filename_queue = tf.train.string_input_producer([data_path], num_epochs=1)

image, label = read(filename_queue)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    img, lab = sess.run([image, label])

    print(img[0,:,:,:].shape)
