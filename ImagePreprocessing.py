''' author: acelik'''


import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import pickle
import csv
import random
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from matplotlib import pyplot as plt

''' IMAGE DATA PREPROCESSING'''

dir = '/Users/aslihancelik/Desktop/Petimages'

class_categories = ['Cat', 'Dog']

data = []

""" Getting the image data"""

for category in class_categories:
    path = os.path.join(dir,category)
    label = class_categories.index(category)  #0 for cat and 1 for dog

    for img in os.listdir(path):
        imgpath = os.path.join(path,img)

        try:
            pet_img = cv2.imread(imgpath,cv2.IMREAD_COLOR)
            pet_img = cv2.resize(pet_img,(128,128))
            image = np.array(pet_img).flatten()

            data.append([image,label])
        except Exception as e:
            pass

random.shuffle(data)
features = []
labels = []


for feature,label in data:
    features.append(feature)
    labels.append(label)
    


"""to get equal amounts of pictures from each classes"""
y=0
for i in range (len(labels)):
    if labels[i]==0 :
        del labels[i]
        del features[i]
        y+=1
    if y==6:
        break
            
#print(len(labels),len(features))

"""to check if there is equal amounts of pictures from two classes"""
yayy=0
mayy=0
for i in range (len(labels)):
    if labels[i]==0 :
        yayy+=1
    else:
        mayy+=1

print(yayy , mayy)

"""getting the first 2000 images from shuffled data"""
features_10 = []
labels_10 = []

say=0
say1=0
sayi=1000      #2 times of this number determines the size of the dataset
for i in range (len(labels)):
        if labels[i]==0 and say<sayi:
            features_10.append(features[i])
            labels_10.append(labels[i])
            say +=1
        elif labels[i]==1 and say1<sayi:
            features_10.append(features[i])
            labels_10.append(labels[i])
            say1 +=1

#print(say,say1)
#print(len(features_10),len(labels_10))

X = np.array(features_10).reshape(-1, 128, 128, 3)
print(X)
print(labels_10)

plt.imshow(X[0])
plt.show()

'''To save sets '''
pick_in = open('/Users/aslihancelik/Desktop/2000_128_feature.pickle', 'wb')
pickle.dump(X,pick_in)
pick_in.close()

pick_in = open('/Users/aslihancelik/Desktop/2000_128_label.pickle', 'wb')
pickle.dump(labels_10,pick_in)
pick_in.close()


'''Showing the Image from Different Datasets'''

#pick = open('/Users/aslihancelik/Desktop/datasets/2000_128_feature.pickle', 'rb')
#features_10= pickle.load(pick)
#pick.close()
#
#X = np.array(features_10).reshape(-1, 128, 128, 3)
#
#
#plt.imshow(X[50])
#plt.show()



