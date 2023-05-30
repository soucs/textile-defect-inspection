import numpy as np
import matplotlib.pyplot as plt
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

img = resize_img(cv2.imread('./dataset/c2_jute/oilstains_colorerror/oilstains_colorerror_01.tif'))
k = 20
avgs = []
for i in range(k,501,k):
    row = []
    for j in range(k,501,k):
        slice = img[i-k:i,j-k:j]
        # avg = np.mean(slice.flatten())
        avg = np.linalg.norm(slice.flatten(),ord=2)
        row.append(avg)
    avgs.append(row)

avgs = np.array(avgs)
print(avgs.shape)

x = np.arange(0,256,5)
fig, ax = plt.subplots(5,5, sharey=True)
i = j= 0
for row in avgs:
    ax[i,j].plot(row)
    j += 1
    if j%5==0:
        i += 1
        j = 0
plt.show()

# show([img,avgs])