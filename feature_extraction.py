import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2 as cv
import h5py

# df_org = pd.read_csv(r'dataset/train64.csv')
# df = df_org[['angle', 'indication_type', 'indication_value']]
file_path = r'/home/soucs/Python/textile-defect-inspection/dataset/textile_defect_data.hdF5'
imgs = h5py.File(file_path)['jute_defect_imgs']
labels = h5py.File(file_path)['jute_defect_labels'][:]

features = {'c1':[],'c2':[],'c3':[],'c4':[],'c5':[],'label':labels}
for img in imgs:
    hist = cv.calcHist([img], [0], None, [256], [0, 256])
    c1 = np.argmax(hist)
    c2 = np.amin(hist[np.nonzero(hist)])
    c3 = np.amax(hist[np.nonzero(hist)])
    c4 = (c3-c2)/255
    c5 = c4/c1
    features['c1'].append(c1)
    features['c2'].append(c2)
    features['c3'].append(c3)
    features['c4'].append(c4)
    features['c5'].append(c5)

hist_features = pd.DataFrame(features)
print(hist_features.label.value_counts())

hist_features.to_csv(r'/home/soucs/Python/textile-defect-inspection/dataset/hist_features.csv', index=False)

# # Viewing images
# cv.imshow('Img',img)
# cv.waitKey(0)
# cv.destroyAllWindows()