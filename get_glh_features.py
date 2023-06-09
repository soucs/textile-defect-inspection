import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2 as cv
import h5py

# label_dict = {'good':0, 'holes_cuts':1, 'threaderror':2, 'oilstains_colorerror':3, 'wrinkles':4, 'foreignbodies':5}
file_path = r'C:\Users\LENOVO\Downloads\textile_defect_data.hdF5'
imgs = h5py.File(file_path)['images'][:]
labels = h5py.File(file_path)['labels'][:]

features = {'c1':[],'c2':[],'c3':[],'c4':[],'c5':[],'c6':[],'label':labels}

def get_c6(hist):
    hist = hist.flatten()
    h_deriv = np.gradient(hist)
    h_deriv2 = np.gradient(h_deriv)
    numer = denom = 0
    for i in range(len(hist)):
        if h_deriv[i]>0:
            numer += h_deriv[i]*hist[i]
        if h_deriv2[i]<0:
            denom += h_deriv[i]*hist[i]
    denom += 1e-20 # To avoid zero division
    div = numer/denom
    return div

for img in imgs:
    hist = cv.calcHist([img], [0], None, [256], [0, 256])
    c1 = np.argmax(hist)
    c2 = np.amin(hist[np.nonzero(hist)])
    c3 = np.amax(hist[np.nonzero(hist)])
    c4 = (c3-c2)/255
    c5 = c4/c1
    c6 = get_c6(hist)
    features['c1'].append(c1)
    features['c2'].append(c2)
    features['c3'].append(c3)
    features['c4'].append(c4)
    features['c5'].append(c5)
    features['c6'].append(c6)

hist_features = pd.DataFrame(features)

hist_features.to_csv(r'C:\Users\LENOVO\CV\textile-defect-inspection\dataset\glh_features.csv', index=False)
print('Feature dataset created')

# # Viewing images
# cv.imshow('Img',img)
# cv.waitKey(0)
# cv.destroyAllWindows()