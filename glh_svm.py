import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.feature_selection import SelectKBest, chi2

# label_dict = {'good':0, 'holes_cuts':1, 'threaderror':2, 'oilstains_colorerror':3, 'wrinkles':4, 'foreignbodies':5}
file_path = r'/home/soucs/Python/textile-defect-inspection/dataset/hist_features.csv'
df = pd.read_csv(file_path)

# X = df.iloc[:,:-1]
X = df[['c4','c6']]
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=None)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


'''# Feature selection using SelectKBest with chi-square test
k = 2  # Number of top features to select
selector = SelectKBest(score_func=chi2, k=k)
X_new = selector.fit_transform(X_train, y_train)

# Get the indices of the selected features
selected_indices = selector.get_support(indices=True)
print(selected_indices)

# Print the indices and names of the selected features
selected_features = [df.columns[i] for i in selected_indices]
print("Selected Features:")
for feature in selected_features:
    print(feature)'''



# # Principal Component Analysis
# y = df['label']
# pca = PCA().fit(X_train)
# Xpca = pca.transform(X_train)
    
# # Explained variance ratio by each principal component
# v_ratio = pca.explained_variance_ratio_
# v_ratio = [v*100 for v in v_ratio]
# print('PCA:',v_ratio)

# pca = PCA(n_components=1)
# X_train, X_test = pca.fit_transform(X_train), pca.transform(X_test)

# SVM Classification
svc = SVC(decision_function_shape='ovo', kernel='rbf',random_state=0)

svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)

print(accuracy_score(y_test, y_pred)) 
print(confusion_matrix(y_test, y_pred))

# Accuracy: 0.35555555555555557; svm with pca; wrinkles prediction better