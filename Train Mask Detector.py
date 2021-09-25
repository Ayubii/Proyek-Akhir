
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.utils import to_categorical

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img

from tensorflow.keras.applications import NASNetMobile
from tensorflow.keras.applications.nasnet import preprocess_input
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import VGG16

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold

from imutils import paths
import os


LR = 1e-5
BS = 32
EP = 50
num_folds = 5


labels = []
data = []


DIR = r"/content/drive/MyDrive/dataset/"
CAT = ['without_mask', 'with_mask']


#For inceptionV3
print("Loading Images")
for cat in CAT:
    path = os.path.join(DIR,cat)
    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        image = load_img(img_path, target_size=(224,224))
        
        image = img_to_array(image)
        image = preprocess_input(image)

        data.append(image)
        labels.append(cat)

print("Images loaded")

#One hot encoding labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

data = np.array(data, dtype="float32")
labels = np.array(labels)

(trainX, testX, trainY, testY) = train_test_split(data,
                                                  labels,
                                                  test_size=0.20,
                                                  stratify=labels,
                                                  random_state=42)
aug = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest")

import tensorflow as tf
from tensorflow import keras
callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', min_delta=0, patience=0, verbose=1, mode='auto',
    baseline=None, restore_best_weights=False
)

kfold = KFold(n_splits=num_folds, shuffle=True)

acc_per_fold = []
loss_per_fold = []
fold_no = 1

for train, test in kfold.split(data, labels):
  baseModel = VGG16(weights="imagenet", 
                          include_top=False,
                          input_tensor=Input(shape=(224,224,3)))

  myModel = baseModel.output
  myModel = AveragePooling2D(pool_size=(7, 7))(myModel)
  myModel = Flatten(name="flatten")(myModel)
  myModel = Dense(128, activation="relu")(myModel)
  myModel = Dense(128, activation="relu")(myModel)
  myModel = Dropout(0.5)(myModel)
  myModel = Dense(2, activation="softmax")(myModel)
  model = Model(inputs=baseModel.input, outputs=myModel)

  for layer in baseModel.layers:
      layer.trainable = False

  opt = Adam(lr=LR, decay=LR/EP)
  model.compile(loss="binary_crossentropy", 
                optimizer=opt,
                metrics=["accuracy"])
  
  print('------------------------------------------------------------------------')
  print(f'Training for fold {fold_no} ...')

  hist = model.fit(
      aug.flow(trainX, trainY, batch_size=BS),
      steps_per_epoch=len(trainX) // BS,
      validation_data=(testX, testY),
      validation_steps=len(testX) // BS,
      epochs=EP,)
  
  scores = model.evaluate(data[test], labels[test], verbose=0)
  print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
  acc_per_fold.append(scores[1] * 100)
  loss_per_fold.append(scores[0])

  # Increase fold number
  fold_no = fold_no + 1

print('------------------------------------------------------------------------')
print('Score per fold')
for i in range(0, len(acc_per_fold)):
  print('------------------------------------------------------------------------')
  print(f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
print('------------------------------------------------------------------------')
print('Average scores for all folds:')
print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
print(f'> Loss: {np.mean(loss_per_fold)}')
print('------------------------------------------------------------------------')

#Eval
predIdxs = model.predict(testX, batch_size=BS)

# for each image in the testing set we need to find the index of the
# label with corresponding largest predicted probability
predIdxs = np.argmax(predIdxs, axis=1)

# show a nicely formatted classification report
print(classification_report(testY.argmax(axis=1), predIdxs,target_names=lb.classes_))

# serialize the model to disk
# model.save(DIR+"mask_detector_mbv2test0ii.model", save_format="h5")



# plot the training loss and accuracy
N = EP
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), hist.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), hist.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), hist.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), hist.history["val_accuracy"], label="val_acc")

tn, fp, fn, tp = confusion_matrix(testY.argmax(axis=1), predIdxs).ravel()
print("True Negatives: ",tn)
print("False Positives: ",fp)
print("False Negatives: ",fn)
print("True Positives: ",tp)



plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
# plt.savefig("vgg16e100-plot.png")