# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
#from telegram_bot.telegram_bot import TelegramBot
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import time
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import keras
from keras import layers as kl

from PIL import Image
from keras.applications import InceptionResNetV2
from tqdm import tqdm
import cv2

import imgaug as ia
from imgaug import augmenters as iaa
import gcsfs
import tensorflow as tf

from class_weights_calculations import get_class_weights

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

fs = gcsfs.GCSFileSystem(project='inlaid-marker-222600')

# Any results you write to the current directory are saved as output.
BATCH_SIZE = 128
SEED = 777
SHAPE = (192, 192, 4)

DATA_DIR = ''
VAL_RATIO = 0.1
THRESHOLD = 0.05

ia.seed(SEED)

def trainDataset():
    path_to_train = DATA_DIR + 'data/'
    data = pd.read_csv(DATA_DIR +'train.csv')
    
    names_with_path = []
    labels = []
    
    for name, label in zip(data['Id'], data['Target'].str.split(' ')):
        y = np.zeros(28) # 28 is the number of the labels
        for lbl in label:
            y[int(lbl)] = 1
            
        names_with_path.append(os.path.join(path_to_train, name))
        labels.append(y)
    return np.array(names_with_path), np.array(labels)

def testDataset():
    path_to_train = DATA_DIR + 'test/'
    data = pd.read_csv(DATA_DIR + 'sample_submission.csv')
    
    names_with_path = []
    labels = []
    
    for name in data['Id']:
        y = np.array(28)
        
        names_with_path.append(os.path.join(path_to_train, name))
        labels.append(y)
    return np.array(names_with_path), np.array(labels)      
    
class ProteinDataGenerator(keras.utils.Sequence):
            
    def __init__(self, paths, labels, batch_size, shape, shuffle = False, use_cache = False, augment = False):
        self.paths, self.labels = paths, labels
        self.batch_size = batch_size
        self.shape = shape
        self.shuffle = shuffle
        self.use_cache = use_cache
        self.augment = augment
        if use_cache == True:
            self.cache = np.zeros((paths.shape[0], shape[0], shape[1], shape[2]), dtype=np.float16)
            self.is_cached = np.zeros((paths.shape[0]))
        self.on_epoch_end()
    
    def __len__(self):
        return int(np.ceil(len(self.paths) / float(self.batch_size)))
    
    def __getitem__(self, idx):
        indexes = self.indexes[idx * self.batch_size : (idx+1) * self.batch_size]

        paths = self.paths[indexes]
        X = np.zeros((paths.shape[0], self.shape[0], self.shape[1], self.shape[2]))
        # Generate data
        if self.use_cache == True:
            X = self.cache[indexes]
            for i, path in enumerate(paths[np.where(self.is_cached[indexes] == 0)]):
                image = self.__load_image(path)
                self.is_cached[indexes[i]] = 1
                self.cache[indexes[i]] = image
                X[i] = image
        else:
            for i, path in enumerate(paths):
                X[i] = self.__load_image(path)

        y = self.labels[indexes]
                
        if self.augment == True:
            seq = iaa.Sequential([
                iaa.OneOf([
                    iaa.Fliplr(0.5), # horizontal flips
                    iaa.Crop(percent=(0, 0.1)), # random crops
                    # Small gaussian blur with random sigma between 0 and 0.5.
                    # But we only blur about 50% of all images.
                    iaa.Sometimes(0.5,
                        iaa.GaussianBlur(sigma=(0, 0.5))
                    ),
                    # Strengthen or weaken the contrast in each image.
                    iaa.ContrastNormalization((0.75, 1.5)),
                    # Add gaussian noise.
                    # For 50% of all images, we sample the noise once per pixel.
                    # For the other 50% of all images, we sample the noise per pixel AND
                    # channel. This can change the color (not only brightness) of the
                    # pixels.
                    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
                    # Make some images brighter and some darker.
                    # In 20% of all cases, we sample the multiplier once per channel,
                    # which can end up changing the color of the images.
                    iaa.Multiply((0.8, 1.2), per_channel=0.2),
                    # Apply affine transformations to each image.
                    # Scale/zoom them, translate/move them, rotate them and shear them.
                    iaa.Affine(
                        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                        rotate=(-180, 180),
                        shear=(-8, 8)
                    )
                ])], random_order=True)

            X = np.concatenate((X, seq.augment_images(X), seq.augment_images(X), seq.augment_images(X)), 0)
            y = np.concatenate((y, y, y, y), 0)
        
        return X, y
    
    def on_epoch_end(self):
        
        # Updates indexes after each epoch
        self.indexes = np.arange(len(self.paths))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __iter__(self):
        """Create a generator that iterate over the Sequence."""
        for item in (self[i] for i in range(len(self))):
            yield item
            
    def __load_image(self, path):
        R = Image.open(path + '_red.png')
        G = Image.open(path + '_green.png')
        B = Image.open(path + '_blue.png')
        Y = Image.open(path + '_yellow.png')

        im = np.stack((
            np.array(R), 
            np.array(G), 
            np.array(B),
            np.array(Y)), -1)
        
        im = cv2.resize(im, (SHAPE[0], SHAPE[1]))
        im = np.divide(im, 255)
        return im


from keras.models import Sequential, load_model
from keras.layers import Dropout, Flatten, Dense, Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D, \
    Activation
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K, Input, Model
import tensorflow as tf

from tensorflow import set_random_seed
set_random_seed(SEED)

def f1_measure(y_true, y_pred):
    y_pred = K.cast(K.greater(K.clip(y_pred, 0, 1), THRESHOLD), K.floatx())
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)

def f1_loss(y_true, y_pred):
    return 1 - f1_measure(y_true, y_pred)


# TODO: try pre trained resnet
def pre_trained_resnet(input_shape, n_out):
    inp = Input(input_shape)
    pretrain_model = InceptionResNetV2(include_top=False, weights=None, input_tensor=inp)
    x = pretrain_model.output
    x = GlobalAveragePooling2D()(x)
    # x = Flatten()(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(1024)(x)
    x = Activation('relu')(x)
    x = Dense(n_out)(x)
    x = Activation('sigmoid')(x)
    for layer in pretrain_model.layers:
        layer.trainable = True
    return Model(inp, x)


def edge_detection_model(input_shape, dropout_rate):
    from keras import backend as K

    def my_init(shape, dtype=None):
        print(shape)
        # np.array([[-3, 0, 3], [-10, 0, +10], [-3, 0, +3]])
        np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])

        return K.random_normal(shape, dtype=dtype)

    inputs = kl.Input(input_shape)
    x = Conv2D(8, (3, 3), activation='relu', input_shape=input_shape, kernel_initializer=my_init, padding='same', trainable=False)(inputs)
    print(inputs.shape)
    print(x.shape)
    x = kl.concatenate([inputs, x])
    x = Conv2D(8, (3, 3), activation='relu')(x)
    x = BatchNormalization(axis=-1)(x)
    x = Conv2D(8, (3, 3), activation='relu')(x)
    x = BatchNormalization(axis=-1)(x)
    x = Conv2D(8, (3, 3), activation='relu')(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(dropout_rate)(x)
    x = Conv2D(16, (5, 5), activation='relu')(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(dropout_rate)(x)
    x = Conv2D(32, (5, 5), activation='relu')(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(dropout_rate)(x)
    x = Conv2D(64, (5, 5), activation='relu')(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(dropout_rate)(x)
    x = Flatten()(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(100, activation='relu')(x)
    x = BatchNormalization(axis=-1)(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(28, activation='sigmoid')(x)
    return keras.Model(inputs=(inputs, ), outputs=(x, ))


def build_model(input_shape, dropout_rate):
    model = Sequential()
    model.add(Conv2D(8, (3, 3), activation='relu', input_shape= input_shape))
    model.add(BatchNormalization(axis=-1))
    model.add(Conv2D(8, (3, 3), activation='relu'))
    model.add(BatchNormalization(axis=-1))
    model.add(Conv2D(16, (3, 3), activation='relu'))
    model.add(BatchNormalization(axis=-1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout_rate))
    #c1 = Conv2D(16, (3, 3), padding='same')(x)
    #c1 = ReLU()(c1)
    #c2 = Conv2D(16, (5, 5), padding='same')(x)
    #c2 = ReLU()(c2)
    #c3 = Conv2D(16, (7, 7), padding='same')(x)
    #c3 = ReLU()(c3)
    #c4 = Conv2D(16, (1, 1), padding='same')(x)
    #c4 = ReLU()(c4)
    #x = Concatenate()([c1, c2, c3, c4])
    model.add(Conv2D(16, (5, 5), activation='relu'))
    model.add(BatchNormalization(axis=-1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout_rate))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(BatchNormalization(axis=-1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout_rate))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(BatchNormalization(axis=-1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout_rate))
    model.add(Flatten())
    model.add(Dropout(dropout_rate))
    model.add(Dense(100, activation='relu'))
    model.add(BatchNormalization(axis=-1))
    model.add(Dropout(dropout_rate))
    model.add(Dense(28, activation='sigmoid'))
    return model


model = edge_detection_model(SHAPE, dropout_rate=0.2)
learning_rate = 1.3e-3

model.compile(
    loss = 'binary_crossentropy',
    optimizer = Adam(learning_rate),
    metrics = [f1_measure]
)

model.summary()

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

# https://keras.io/callbacks/#modelcheckpoint
checkpoint = ModelCheckpoint('./base.model', monitor='val_f1_measure', verbose=1, save_best_only=True, save_weights_only=False, mode='min', period=1)
earlystopper = EarlyStopping(monitor='val_f1_measure', patience=15, verbose=1,mode='max')
reduceLROnPlato = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1, mode='min')

# some params
epochs = 500
train_model = True
save_model = True
#bot = TelegramBot()
###
if train_model:

    #bot.send_message('Training started')
    class_weigths = get_class_weights()
    history = model.fit_generator( generator = train_gen, steps_per_epoch = len(train_gen),validation_data = valid_gen, validation_steps = 8,
        epochs = epochs, verbose = 1, callbacks = [checkpoint,earlystopper], class_weight=get_class_weights())
    #bot.send_message('Training ended')

    if save_model:
        ts = int(time.time())
        model.save('my_model'+str(ts)+'.h5')
        print("Model's weights saved!")



# history.history

# bestModel = load_model('./base.model', custom_objects={'f1': f1}) #, 'f1_loss': f1_loss})
def evaluate():
    print("Loading weights ...")
    import glob
    import os

    list_of_files = glob.glob('*.h5')  # * means all if need specific format then *.csv
    latest_file = max(list_of_files, key=os.path.getctime)
    model = load_model(latest_file, custom_objects={'f1_measure': f1_measure})
    fullValGen = valid_gen

    lastFullValPred = np.empty((0, 28))
    lastFullValLabels = np.empty((0, 28))
    for i in tqdm(range(len(fullValGen))):
        im, lbl = fullValGen[i]
        scores = model.predict(im)
        lastFullValPred = np.append(lastFullValPred, scores, axis=0)
        lastFullValLabels = np.append(lastFullValLabels, lbl, axis=0)
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
        score = model.predict(images)
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

#evaluate()
