# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
#from telegram_bot.telegram_bot import TelegramBot
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import keras

from PIL import Image
from tqdm import tqdm
import cv2

import imgaug as ia
from imgaug import augmenters as iaa
import gcsfs
import tensorflow as tf
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

def trainDataset():
    '''
    In order to speed up the process this trainDataset function used in the baseline approach only watches at labels
    :return:
    '''
    data = pd.read_csv('train.csv')
    labels = []
    
    for name, label in zip(data['Id'], data['Target'].str.split(' ')):
        y = np.zeros(28) # 28 is the number of the labels
        for lbl in label:
            y[int(lbl)] = 1
        labels.append(y)
    return np.array(labels)

def testDataset():
    data = pd.read_csv('sample_submission.csv')

    names_with_path = []
    labels = []
    
    for name in data['Id']:
        y = np.array(28)
        labels.append(y)
    return np.array(labels)


from sklearn.metrics import f1_score as off1
rng = np.arange(0, 1, 0.001)
f1s = np.zeros((rng.shape[0], 28))
for j,t in enumerate(tqdm(rng)):
    for i in range(28):
        p = np.array(lastFullValPred[:,i]>t, dtype=np.int8)
        scoref1 = off1(lastFullValLabels[:,i], p, average='binary')
        f1s[j,i] = scoref1
        
print('Individual F1-scores for each class:')
print(np.max(f1s, axis=0))
print('Macro F1-score CV =', np.mean(np.max(f1s, axis=0)))

T = np.empty(28)
for i in range(28):
    T[i] = rng[np.where(f1s[:,i] == np.max(f1s[:,i]))[0][0]]
print('Probability threshold maximizing CV F1-score for each class:')
print(T)

pathsTest, labelsTest = testDataset()

testg = ProteinDataGenerator(pathsTest, labelsTest, BATCH_SIZE, SHAPE)
submit = pd.read_csv(DATA_DIR + 'sample_submission.csv')
P = np.zeros((pathsTest.shape[0], 28))
for i in tqdm(range(len(testg))):
    images, labels = testg[i]
    score = model.predict(images)
    P[i*BATCH_SIZE:i*BATCH_SIZE+score.shape[0]] = score
    
PP = np.array(P)
prediction = []

for row in tqdm(range(submit.shape[0])):
    
    str_label = ''
    
    for col in range(PP.shape[1]):
        if(PP[row, col] < T[col]):
            str_label += ''
        else:
            str_label += str(col) + ' '
    prediction.append(str_label.strip())

submit['Predicted'] = np.array(prediction)
submit.to_csv('baseline2.csv', index=False)
#bot.send_message('Program terminated correctly')
