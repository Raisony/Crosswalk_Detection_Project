import PIL
import cv2, os
import imutils
import matplotlib
import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt

IMG_HEIGHT, IMG_WIDTH, CHANNEL = 800, 800, 3
TRAINING_IMG = './Trainingset/Image/'
TRAINING_MSK = './Trainingset/Label/'

Traingset_img = sorted(os.listdir(TRAINING_IMG), key = None)[200:700] 
Traingset_msk = sorted(os.listdir(TRAINING_MSK), key = None)[200:700] 

# Noise
def aug_img(rgbImg):
    augmented_images = []
    augmented_images.append(rgbImg)
    gaussian_noise = rgbImg.copy()
    g1 = cv2.randn(gaussian_noise, 0, 100)
    augmented_images.append(rgbImg + g1)
    uniform_noise = rgbImg.copy()
    u1 = cv2.randu(uniform_noise, 0, 1)
    augmented_images.append(rgbImg + u1)
    negmask = np.where(rgbImg > 200)
    aug = rgbImg.copy()
    aug[negmask] -= 20
    augmented_images.append(aug)
    return augmented_images

for i, e in enumerate(TRAINING_IMG):
    img = cv2.imread('./Image/' + e)
    augmentation = aug_img(img)
    for num, img in enumerate(augmentation):
        matplotlib.image.imsave('./Image_aug/Img{}{}.png'.format(i, num), img)
        mask = cv2.imread('./Label/' + Traingset_msk[i])
        matplotlib.image.imsave('./Label_aug/Lab{}{}.png'.format(i, num), mask)
    print(i,e)

# Rotation
L1, L2 = os.listdir('./Train_aug/Image/'), os.listdir('./Train_aug/Label/')

for i, e in enumerate(L1):
    img = np.asarray(PIL.Image.open('./Train_aug/Image/' + e))
    lab = np.asarray(PIL.Image.open('./Train_aug/Label/' + L2[i]))
    rotated_img = imutils.rotate(img, 30*i)
    rotated_lab = imutils.rotate(lab, 30*i)
    matplotlib.image.imsave('./Trainingset/Image/Img{}.jpg'.format(i), rotated_img)
    matplotlib.image.imsave('./Trainingset/Label/Lab{}.jpg'.format(i), rotated_lab)
    print(i)

