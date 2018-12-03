from keras.engine.saving import load_model
from Project.CNN import ProteinDataGenerator, trainDataset, testDataset

import os
import time
from tqdm import tqdm
import glob
import numpy as np
import pandas as pd
"""
print("Loading weights ...")
list_of_files = glob.glob('*.h5')  # * means all if need specific format then *.csv
latest_file = max(list_of_files, key=os.path.getctime)
model = load_model(latest_file, custom_objects={'f1_measure': f1_measure})
"""

# history.history
# bestModel = load_model('./base.model', custom_objects={'f1': f1}) #, 'f1_loss': f1_loss})
BATCH_SIZE = 128
SEED = 777
SHAPE = (192, 192, 4)
DATA_DIR = ''
VAL_RATIO = 0.1
THRESHOLD = 0.05

paths, labels = trainDataset()

keys = np.arange(paths.shape[0], dtype=np.int)
np.random.seed(SEED)
np.random.shuffle(keys)

# divide train dataset into validation and training set
lastTrainIndex = int((1-VAL_RATIO) * paths.shape[0])

pathsTrain = paths[0:lastTrainIndex]
labelsTrain = labels[0:lastTrainIndex]

pathsVal = paths[lastTrainIndex:]
labelsVal = labels[lastTrainIndex:]

print(paths.shape, labels.shape)
print(pathsTrain.shape, labelsTrain.shape, pathsVal.shape, labelsVal.shape)

train_gen = ProteinDataGenerator(pathsTrain, labelsTrain, BATCH_SIZE, SHAPE, use_cache=True, augment = False, shuffle = False)
valid_gen = ProteinDataGenerator(pathsVal, labelsVal, BATCH_SIZE, SHAPE, use_cache=True, shuffle = False)
fullValGen = valid_gen

lastFullValPred = np.empty((0, 28))
lastFullValLabels = np.empty((0, 28))

for i in tqdm(range(len(fullValGen))):
    im, lbl = fullValGen[i]
    lastFullValLabels = np.append(lastFullValLabels, lbl, axis=0)
#In the following part we set the prediction to be 1 only on the first class that is Nucleus.
lastFullValPred = np.zeros(lastFullValLabels.shape)
lastFullValPred[:,0] = 1
print(lastFullValPred.shape, lastFullValLabels.shape)

from sklearn.metrics import f1_score as off1, f1_score

rng = np.arange(0, 1, 0.001)
f1s = np.zeros((rng.shape[0], 28))
for j, t in enumerate(tqdm(rng)):
    for i in range(28):
        p = np.array(lastFullValPred[:, i] > t, dtype=np.int8)
        scoref1 = off1(lastFullValLabels[:, i], p, average='binary')
        f1s[j, i] = scoref1

print('Individual F1-scores for each class:')
print(np.max(f1s, axis=0))
print('Macro F1-score CV =', np.mean(np.max(f1s, axis=0)))

T = np.empty(28)
for i in range(28):
    T[i] = rng[np.where(f1s[:, i] == np.max(f1s[:, i]))[0][0]]
print('Probability threshold maximizing CV F1-score for each class:')
print(T)

pathsTest, labelsTest = testDataset()
testg = ProteinDataGenerator(pathsTest, labelsTest, BATCH_SIZE, SHAPE)
submit = pd.read_csv(DATA_DIR + 'sample_submission.csv')
P = np.zeros((pathsTest.shape[0], 28))



for i in tqdm(range(len(testg))):
    images, labels = testg[i]
    score = np.zeros((36, 28))
    score[:, 0] = 1
    P[i * BATCH_SIZE:i * BATCH_SIZE + score.shape[0]] = score

PP = np.array(P)
prediction = []

for row in tqdm(range(submit.shape[0])):

    str_label = ''

    for col in range(PP.shape[1]):
        if (PP[row, col] < T[col]):
            str_label += ''
        else:
            str_label += str(col) + ' '
    prediction.append(str_label.strip())

submit['Predicted'] = np.array(prediction)
ts = str(int(time.time()))
submit.to_csv('model' + ts + '.csv', index=False)
#bot.send_message('Program terminated correctly')
