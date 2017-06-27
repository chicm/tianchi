import numpy as np
import os, glob
import cv2
import settings

working_dir = settings.BASE_DIR+'tutorial_out/'
img_dir = working_dir + 'imgs/'

filenames = glob.glob(working_dir+'masks*.npy')

num = 0
for f in filenames:
    img = np.load(f)
    print(img.shape) 
    num += 1
    for i in range(img.shape[0]):
        fn = working_dir+'masks/'+str(num)+str(i)+'.jpg'
        print(fn)
        cv2.imwrite(fn, img[i]*255)
        
train_imgs = np.load(working_dir+'trainMasks.npy')
print(train_imgs.shape)

for i in range(train_imgs.shape[0]):
    img = train_imgs[i, 0] #* 255
    #print(img.shape)
    #print(img)
    fn = img_dir+str(i)+'_m.jpg'
    #print(fn)
    #cv2.imwrite(fn, img)
