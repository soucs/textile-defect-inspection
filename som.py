import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn_som.som import SOM

import h5py
from statistics import mode
import cv2

def resize_img(img):
    dim = (500, 500)
    return cv2.resize(img,dim,interpolation=cv2.INTER_AREA)
def show(imgs):
    try:
        for i, img in enumerate(imgs):
            cv2.imshow('img'+str(i),img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except:
        print('Invalid format to display')
def blur(img):
    ksize = (5,5)
    return cv2.blur(img,ksize)

file_path = r'/home/soucs/Python/textile-defect-inspection/dataset/textile_defect_data.hdF5'
imgs = h5py.File(file_path)['images'][:]
labels = h5py.File(file_path)['labels'][:]
fabric = h5py.File(file_path)['fabric'][:]

i = 999
org_img, lab = resize_img(imgs[i]), labels[i]
img = blur(org_img.copy())

kernel = 5
part_img = []
for r in range(0,500,kernel):
    for c in range(0,500,kernel):
        part = img[r:r+kernel,c:c+kernel].flatten()
        part_img.append(part)

part_img = np.array(part_img)
m = 3; n = 1
clust = SOM(m=m, n=n, dim=kernel**2)
clust_img = clust.fit_predict(part_img)
print(set(clust_img.tolist()))

k = 0
shade = int(255/(m*n))
for r in range(0,500,kernel):
    for c in range(0,500,kernel):
        for pi,p in enumerate(range(0,255,shade)):
            if clust_img[k] == pi:
                img[r:r+kernel,c:c+kernel] = np.full(shape=(kernel,kernel), fill_value=p)
        # if clust_img[k] == 0:
        #     img[r:r+kernel,c:c+kernel] = np.zeros_like(img[r:r+kernel,c:c+kernel])
        # else:
        #     img[r:r+kernel,c:c+kernel] = np.full(shape=(kernel,kernel), fill_value=255)
        k += 1

show([org_img,img])
# show([img,clust_img])
# print(lab)

# show([img,g_img])

# clust = DBSCAN()
# clust_img = clust.fit