'''Outlier Scenario'''

import time
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras import Input, optimizers
from tensorflow.python.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.python.keras.models import Model
from tensorflow.keras.optimizers import SGD

# batch_size = 32
size_small_height = 64
size_small_width = 64
size_large_height = 128
size_large_width =128

test_data = tf.keras.preprocessing.image_dataset_from_directory(
  '/Users/aslihancelik/Desktop/outlier',
  seed=0,
  image_size=(size_large_height, size_large_width)
)

model = tf.keras.models.load_model('/Users/aslihancelik/Desktop/models/model_4')
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

test_loss, test_acc = model.evaluate(test_data)
print('\nTest accuracy:', test_acc)

test_start = time.time()
predictions = model.predict(test_data)
test_end = time.time()

print(predictions)
print('total testing time: {0}'.format(test_end - test_start))
# # model.save('/Users/aslihancelik/Desktop/models/outlier')



