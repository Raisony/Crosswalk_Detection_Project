import cv2, os
import numpy as np
from Unet_model import Unet
from keras import optimizers
from skimage.transform import resize
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from skimage.io import imread, imshow, imread_collection, concatenate_images
from metrics import iouCalculate, iouThresholdedCalculate, F1ScoreCalculate
IMG_HEIGHT, IMG_WIDTH, CHANNEL = 800, 800, 3
model = Unet(input_shape = (IMG_HEIGHT, IMG_WIDTH, CHANNEL))
adam = optimizers.Adam(lr = 1e-4, beta_1 = 0.95, beta_2 = 0.999, epsilon = None, decay = 0.0, amsgrad = False)
model.compile(optimizer = adam, loss = 'binary_crossentropy', metrics = [iouCalculate, iouThresholdedCalculate, F1ScoreCalculate])
TRAINING_IMG = './Trainingset/Image/'
TRAINING_MSK = './Trainingset/Label/'

# Trainset_img = sorted(os.listdir('./Data/Image/'), key=lambda x: int(x[:-4]))[:100]
# Trainset_msk = sorted(os.listdir('./Data/Mask/'), key=lambda x: int(x[:-4]))[:100]
Trainset_img = sorted(os.listdir(TRAINING_IMG), key=None)[:700] 
Trainset_msk = sorted(os.listdir(TRAINING_MSK), key=None)[:700] 
for i,e in enumerate(Trainset_img):
    if (e[:-4][3:]) != (Trainset_msk[i][:-4][3:]):
        print(e, Trainset_msk[i], 'MISS MATCH')

T = len(Trainset_img)
X_train = np.zeros((T, IMG_HEIGHT, IMG_WIDTH, CHANNEL), dtype = np.float32)
for n, id_ in enumerate(Trainset_img):
    img = imread(TRAINING_IMG + id_)
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode = 'constant', preserve_range = True)
    X_train[n] = img
    if n%100 == 0: print(n)
n, id_ = 0,0

Y_train = np.zeros((T, IMG_HEIGHT, IMG_WIDTH, 1), dtype = np.float32)
for n, id_ in enumerate(Trainset_msk):
    mask = imread(TRAINING_MSK + id_)[:,:,:1]
    mask = resize(mask, (IMG_HEIGHT, IMG_WIDTH), mode ='constant', preserve_range = True)
    mask_ = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype = np.float32)
    mask = np.maximum(mask, mask_)
    L, l = np.where(mask > 0), np.where(mask <= 0)
    mask[L],  mask[l] = True, False
    Y_train[n] = mask
    if n%100 == 0: print(n)

X_train = np.asarray(X_train, dtype = np.float32)/ X_train.max()
Y_train = np.asarray(Y_train, dtype = np.float32)

checkpointer = ModelCheckpoint('new_test.h5', verbose=1, monitor='val_loss', save_best_only=True)
result = model.fit(X_train, Y_train, validation_split=0.1, batch_size=2, epochs=100, callbacks = [checkpointer])
