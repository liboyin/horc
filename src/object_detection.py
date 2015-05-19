__author__ = 'manabchetia'

import cv2
import numpy as np
from matplotlib import pyplot as plt

img_dir = '../data/uni/'

if __name__ == '__main__':
    img = cv2.imread(img_dir+'image001.JPG')

    mask = np.zeros(img.shape[:2],np.uint8)
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)

    rect = (50,50,450,290)
    cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)

    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    img = img*mask2[:,:,np.newaxis]
    plt.imshow(img),plt.colorbar(),plt.show()

    # https://machinelearning1.wordpress.com/2014/08/05/simple-object-detection-in-3-lines-of-code-opencvpython/
    # http://www.pyimagesearch.com/2014/04/21/building-pokedex-python-finding-game-boy-screen-step-4-6/
    # http://docs.opencv.org/master/d8/d83/tutorial_py_grabcut.html