import numpy as np
import h5py

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

# label_dict = {'good':0, 'holes_cuts':1, 'threaderror':2, 'oilstains_colorerror':3, 'wrinkles':4, 'foreignbodies':5}
file_path = r'/home/soucs/Python/textile-defect-inspection/dataset/textile_defect_data.hdF5'
images = h5py.File(file_path)['jute_defect_imgs'][:]
labels = h5py.File(file_path)['jute_defect_labels'][:]

comx_features = np.load('/home/soucs/Python/textile-defect-inspection/dataset/glcm_features.npy')

X_train, X_test, y_train, y_test = train_test_split(comx_features,labels,test_size=0.3,random_state=None)

svc = SVC(decision_function_shape='ovo', kernel='rbf', random_state = None)
svc.fit(X_train,y_train)
y_pred = svc.predict(X_test)
print(accuracy_score(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))