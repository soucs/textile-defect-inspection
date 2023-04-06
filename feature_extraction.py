import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2 as cv
import h5py

df_org = pd.read_csv(r'dataset/train64.csv')
df = df_org[['angle', 'indication_type', 'indication_value']]
imgs = h5py.File(r'dataset/train64.h5')['images']

new_cols = {'c1':[],'c2':[],'c3':[],'c4':[],'c5':[]}
for img in imgs:
    img = np.reshape(img, (64,64))
    img = cv.normalize(img, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
    hist = cv.calcHist([img], [0], None, [256], [0, 256])
    c1 = np.argmax(hist)
    c2 = np.amin(hist[np.nonzero(hist)])
    c3 = np.amax(hist[np.nonzero(hist)])
    c4 = (c3-c2)/255
    c5 = c4/c1
    new_cols['c1'].append(c1)
    new_cols['c2'].append(c2)
    new_cols['c3'].append(c3)
    new_cols['c4'].append(c4)
    new_cols['c5'].append(c5)

features = pd.DataFrame(new_cols)
df_final = pd.concat([df, features], axis=1)

df_final.to_csv(r'/home/soucs/Python/textile-defect-inspection/dataset/textile_defects.csv', index=False)


# plt.plot(hist)
# plt.show()




# # Viewing images
# cv.imshow('Img1',img)
# cv.waitKey(0)
# cv.destroyAllWindows()