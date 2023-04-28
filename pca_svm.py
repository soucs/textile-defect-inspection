import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# label_dict = {'good':0, 'holes_cuts':1, 'threaderror':2, 'oilstains_colorerror':3, 'wrinkles':4, 'foreignbodies':5}
file_path = r'/home/soucs/Python/textile-defect-inspection/dataset/hist_features.csv'
df = pd.read_csv(file_path)

X = df[['c1', 'c2', 'c3', 'c4', 'c5']]
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=1)

# # Principal Component Analysis
# pca = PCA().fit(X_train)
# Xpca = pca.transform(X_train)
    
# # Explained variance ratio by each principal component
# v_ratio = pca.explained_variance_ratio_
# v_ratio = [v*100 for v in v_ratio]
# print('PCA:',v_ratio)

# pca = PCA(n_components=1)
# X_train, X_test = pca.fit_transform(X_train), pca.transform(X_test)

# SVM Classification
svc = SVC(decision_function_shape='ovo', kernel='rbf')

svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)

print(accuracy_score(y_test, y_pred)) 
print(confusion_matrix(y_test, y_pred))

# Accuracy: 0.24666666666666667; svm; all getting predicted as good
# Accuracy: 0.35555555555555557; svm with pca; wrinkles prediction better