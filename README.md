import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import os
from skimage.io import imread
from skimage.transform import resize

plt.figure(figsize=(10,5))
img1 = r"C:\test_set\dogs\dog.4040.jpg"
plt.imshow(imread(img1))

img_path = r"C:\test_set\dogs\dog.4040.jpg"
img = imread(img_path)
img

img.shape

img_resize = resize(img, (15,15))
img_resize.shape

img_resize

flatten_img = img_resize.flatten()
flatten_img

flatten_img.shape

input_dir = r"C:\test_set"
categories = ['cats', 'dogs']
data = []
labels = []
for category_idx, category in enumerate(categories):
    for file in os.listdir(os.path.join(input_dir, category)):
        img_path = os.path.join(input_dir, category, file)
        print(img_path)
        img = imread(img_path)
        img = resize(img, (15,15))
        data.append(img.flatten())
        labels.append(category_idx)    

data[1]

labels[1]

data = np.asarray(data)
labels = np.asarray(labels)

data

labels

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

from sklearn.svm import SVC
svm_model = SVC()

svm_model.fit(x_train, y_train)

y_pred = svm_model.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
score = accuracy_score(y_test, y_pred)
score

from sklearn.model_selection import cross_val_score
cross_val_score = cross_val_score(svm_model, data, labels, cv = 5)
cross_val_score

Mean_Accuracy = cross_val_score.mean()
Mean_Accuracy

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

from sklearn.model_selection import GridSearchCV

classifer = SVC()
parameters = [{'gamma':[0.01, 0.001, 0.0001], 'C':[10, 100, 1000]}]
grid_search = GridSearchCV(classifer, parameters)
grid_search.fit(x_train, y_train)

best_estimator = grid_search.best_estimator_
best_estimator

y_prediction = best_estimator.predict(x_test)
y_prediction

plt.figure(figsize=(10, 5))
img_path = r"C:\test_set\dogs\dog.4040.jpg"
plt.imshow(imread(img1))

import cv2 as cv
img_path = cv.imread(img1)
plt.imshow(img_path)

img_dog = r"C:\test_set\dogs\dog.4040.jpg"
img_path = cv.imread(img_dog)
plt.imshow(img_path)

img_dog = r"C:\test_set\dogs\dog.4040.jpg"
img_new = imread(img_dog)
img_new1 = resize(img_new, (15,15))
img_flatten = img_new1.flatten()
img_array = np.asarray(img_flatten)

result = svm_model.predict(img_array.reshape(1, -1))

if result[0] == 1:
    print("Result =", result[0])
    print("It is a cat.")
else:
    print("It is a dog.")

def image_classification_prediction(image):
    img_new = imread(image)
    img_new1 = resize(img_new, (15,15))
    img_flatten = img_new1.flatten()
    img_array = np.asarray(img_flatten)
    result = svm_model.predict(img_array.reshape(1, -1))
    img_path = cv.imread(image)
    plt.imshow(img_path)
    if result[0] == 1:
        print("Result =", result[0])
        return "It is a cat"
    else:
        return "IT is a dog"

img2 = r"C:\test_set\dogs\dog.5000.jpg"
image_classification_prediction(img2)

img3 = r"C:\test_set\cats\cat.4041.jpg"
image_classification_prediction(img3)










    





































































