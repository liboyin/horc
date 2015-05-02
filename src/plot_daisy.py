__author__ = 'manabchetia'

from skimage.feature import daisy
from PIL import Image
from skimage.io import imread
from skimage import data
# import matplotlib.pyplot as plt
from skimage.color import rgb2gray


img = rgb2gray(imread('../data/uni/IMAGE001.jpg'))



descs, descs_img = daisy(img, step=180, radius=58, rings=2, histograms=6,
                         orientations=8, visualize=True)

print(descs[0])
# fig, ax = plt.subplots()
# ax.axis('off')
# ax.imshow(descs_img)
# descs_num = descs.shape[0] * descs.shape[1]
# ax.set_title('%i DAISY descriptors extracted:' % descs_num)
# plt.show()
