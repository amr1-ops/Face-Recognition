import os
import numpy as np
from sklearn.model_selection import train_test_split
import cv2
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras import layers,models

DATADIR = "dataset"

CATEGORIES = ["Ahmed Amr","Bebo","Hossam Ali","Mohamed Labib","Mohamed Mokhtar"]

data = []


def create_data():
    for category in CATEGORIES:  # do

        path = os.path.join(DATADIR,category)  # create path 
        class_num = CATEGORIES.index(category)  # get the classification

        for img in tqdm(os.listdir(path)):  # iterate over each image
            try:
                img_array = cv2.imread(os.path.join(path,img))  # convert to array
                img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB) #it may be not necessory but i put it as a precaution
                img_array=cv2.resize(img_array, (170, 170))
                img_array = np.array(img_array)
                data.append([img_array, class_num])  # add this to our training_data
            except Exception as e:  # in the interest in keeping the output clean...
                pass
       


create_data()


y = []
features_data = []



for feature,label in data:
    y.append(label)
    features_data.append(feature)  



features_data = np.array(features_data)


print("Features loaded..\n")



Xtrain, Xtest, ytrain, ytest = train_test_split(features_data,
                                                    y,
                                                    test_size=.25,
                                                    random_state=1234123)


Xtrain = np.array(Xtrain)
Xtest = np.array(Xtest)
ytrain = np.array(ytrain)
ytest = np.array(ytest)

print("Start training..\n")

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3),activation='relu', input_shape=(170,170,3)))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Conv2D(64, (3, 3),activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

#--------------------------------------------

model.add(layers.Dropout(0.1))

model.add(layers.Flatten())  # this converts our 3D feature(width,height,channel) maps to 1D feature vectors

model.add(layers.Dense(64,activation='relu'))#feature selection

model.add(layers.Dense(5))


model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])#combination between two GD methodologies "adam"


history = model.fit(Xtrain, ytrain,epochs=15, validation_data=(Xtest, ytest))
test_loss, test_acc = model.evaluate(Xtest,  ytest, verbose=2)
print("accuracy ",test_acc)

#model.save("Face recognition.model")

import matplotlib.pyplot as plt

#loss curve
loss_train = history.history['loss']
epochs = range(1,16)
plt.plot(epochs, loss_train, 'g', label='loss')
plt.title('loss curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

#accuracy curve
loss_train = history.history['accuracy']
epochs = range(1,16)
plt.plot(epochs, loss_train, 'g', label='accuracy')
plt.title('accuracy curve')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd

prediction = model.predict(Xtest).argmax(axis=1)
cm = confusion_matrix(ytest, prediction)

cm_df = pd.DataFrame(cm,
                     index = ["Ahmed Amr","Bebo","Hossam Ali","Mohamed Labib","Mohamed Mokhtar"], 
                     columns = ["Ahmed Amr","Bebo","Hossam Ali","Mohamed Labib","Mohamed Mokhtar"])

plt.figure(figsize=(5,4))
sns.heatmap(cm_df, annot=True)
plt.title('Confusion Matrix')
plt.ylabel('Actal Values')
plt.xlabel('Predicted Values')
plt.show()

#ROC curve 
from sklearn.metrics import roc_curve, auc ,roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from itertools import cycle

fpr = dict()
tpr = dict()
roc_auc = dict()

# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(5)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(5):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= 5

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt.figure()
plt.plot(
    fpr["micro"],
    tpr["micro"],
    label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc["micro"]),
    color="deeppink",
    linestyle=":",
    linewidth=4,
)

plt.plot(
    fpr["macro"],
    tpr["macro"],
    label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc["macro"]),
    color="navy",
    linestyle=":",
    linewidth=4,
)

lw=2

colors = cycle(["aqua", "darkorange", "cornflowerblue"])
for i, color in zip(range(5), colors):
    plt.plot(
        fpr[i],
        tpr[i],
        color=color,
        lw=lw,
        label="ROC curve of class {0} (area = {1:0.2f})".format(i, roc_auc[i]),
    )

plt.plot([0, 1], [0, 1], "k--", lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Some extension of Receiver operating characteristic to multiclass")
plt.legend(loc="lower right")
plt.show()