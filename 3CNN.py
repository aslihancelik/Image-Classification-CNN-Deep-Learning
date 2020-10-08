''' author: acelik '''

import time
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from tensorflow.keras import optimizers
from keras import backend as K
from tensorflow.python.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Activation, Dropout
import pickle
import numpy as np


pick = open('/Users/aslihancelik/Desktop/datasets/1000_64_feature.pickle', 'rb')
feature= pickle.load(pick)
pick.close()

pick = open('/Users/aslihancelik/Desktop/datasets/1000_64_label.pickle', 'rb')
label= pickle.load(pick)
pick.close()

feature = feature/255.0
label = np.array(label)

batch_size = 32

# img_width, img_height = 64, 64

'''1st Model: The Model Used in the Report'''
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=feature.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))


model.compile(loss='binary_crossentropy',
#               optimizer='adam',
              optimizer='rmsprop',
              metrics=['accuracy'])

train_start = time.time()
output = model.fit(feature,label, epochs=20, validation_split=0.2, batch_size=batch_size)
train_end = time.time()

print(train_end-train_start)

'''To See the Model Performance'''

# from sklearn.model_selection import train_test_split
# x_train, x_valid, y_train, y_valid = train_test_split(feature, label, test_size=0.2)

# test_loss, test_acc = model.evaluate(x_train, y_train)
# print('\nTest accuracy:', test_acc)

test_loss, test_acc = model.evaluate(feature, label)
print('\nTest accuracy:', test_acc)

test_start = time.time()
predictions = model.predict(feature)
# predictions = model.predict(x_valid)
test_end = time.time()

print(predictions)
print('total training time: {0}'.format(train_end - train_start))
print('total testing time: {0}'.format(test_end - test_start))
# model.save('/Users/aslihancelik/Desktop/models/model_2')

# summary of history for accuracy
plt.plot(output.history['accuracy'])
plt.plot(output.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['accuracy', 'val_accuracy'], loc='upper left')
plt.show()
# summary of history for loss
plt.plot(output.history['loss'])
plt.plot(output.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['loss', 'val_loss'], loc='upper left')
plt.show()



'''2nd model'''
# model = Sequential()
# model.add(Conv2D(64, (3, 3), input_shape=feature.shape[1:]))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Conv2D(64, (3, 3)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Flatten())

# model.add(Dense(64))
# model.add(Activation('relu'))

# model.add(Dense(1))
# model.add(Activation('sigmoid'))

# model.compile(loss='binary_crossentropy',
#               optimizer='adam',
#               metrics=['accuracy'])

'''3rd model'''
# model = Sequential([

#   Dense(128, activation='relu'),
#   Dense(2, activation='softmax')
# ])
# model.compile(optimizer=optimizers.Adam(lr=0.0001),
#               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#               metrics=['accuracy'])


'''4th model'''

# model = Sequential()
# model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=feature.shape[1:]))
# model.add(MaxPooling2D((2, 2)))
# model.add(Flatten())
# model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
# model.add(Dense(1, activation='sigmoid'))
# # compile model
# opt = SGD(lr=0.001, momentum=0.9)
# model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

# model.compile(loss='binary_crossentropy',
# #               optimizer='adam',
#               optimizer='rmsprop',
#               metrics=['accuracy'])



