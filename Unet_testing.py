import cv2, os
import numpy as np
import matplotlib
from Unet_model import Unet
from keras import optimizers
from skimage.transform import resize
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from skimage.io import imread, imshow, imread_collection, concatenate_images
from metrics import iouCalculate, iouThresholdedCalculate, F1ScoreCalculate
IMG_HEIGHT, IMG_WIDTH, CHANNEL = 800, 800, 3
MODEL = 'new_test.h5'
model = Unet(input_shape = (IMG_HEIGHT, IMG_WIDTH, CHANNEL))
adam = optimizers.Adam(lr = 1e-4, beta_1 = 0.95, beta_2 = 0.999, epsilon = None, decay = 0.0, amsgrad = False)
model.compile(optimizer = adam, loss = 'binary_crossentropy', metrics = [iouCalculate, iouThresholdedCalculate, F1ScoreCalculate])
model.load_weights(MODEL)

TESTING_IMG = './test/'
Testset = os.listdir(TESTING_IMG)

T = len(Testset)
X_test = np.zeros((T, IMG_HEIGHT, IMG_WIDTH, CHANNEL), dtype = np.float32)
for n, id_ in enumerate(Testset):
    img = imread(TESTING_IMG + id_)
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode = 'constant', preserve_range = True)
    X_test[n] = img
    if n%10 == 0: print(n)
X_test = np.asarray(X_test, dtype = np.float32)/ X_test.max()

preds_test = model.predict(X_test, batch_size = 1, verbose = 1)
preds_test_threshold = (preds_test > 0.5).astype(np.float32)

for i in range(T):
    imshow(np.squeeze(X_test[i]))
    plt.show()
    imshow(np.squeeze(preds_test[i]))
    plt.show()
    imshow(np.squeeze(preds_test_t[i]))
    plt.show()


    # '''Model_list = os.listdir('./Model/')
# print(Model_list)
# Pred = []
# for model in ['model_color_retestv6.h5''''

# earlystopper = EarlyStopping(patience=5, verbose=1)
# model = load_model('model_color_150.h5')
# checkpointer = ModelCheckpoint('model_color_2000.h5', verbose=1, save_best_only=True)
# result = model.fit(X_train, Y_train, validation_split=0.1, batch_size=2, epochs=500, callbacks = [checkpointer])
#     print(mod)
model = load_model('model_color_retestv6.h5')
# preds_train = model.predict(X_train[:int(X_train.shape[0]*0.9)], batch_size=1, verbose=1)
# preds_val = model.predict(X_train[int(X_train.shape[0]*0.9):], batch_size=1, verbose=1)
# print(mod)
preds_test = model.predict(X_test,batch_size = 10,verbose = 1)

# Threshold predictions
#
#preds_train_t = (preds_train > 0.5).astype(np.uint8)
#reds_val_t = (preds_val > 0.5).astype(np.uint8)
preds_test_t = (preds_test > 0.2).astype(np.uint8)
for i in range(101):
    imshow(np.squeeze(X_test[i]))
    plt.show()
    imshow(np.squeeze(preds_test[i]))
    plt.show()
    imshow(np.squeeze(preds_test_t[i]))
    plt.show()
# Create list of upsampled test masks
# preds_test_upsampled = []
# for i in range(len(preds_test)):
#     preds_test_upsampled.append(resize(np.squeeze(preds_test[i]),  (sizes_test[i][0], sizes_test[i][1]), mode='constant', preserve_range=True))