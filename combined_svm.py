import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

glh_df = pd.read_csv(r'C:\Users\LENOVO\CV\textile-defect-inspection\dataset\glh_features.csv')
glcm_df = pd.read_csv(r'C:\Users\LENOVO\CV\textile-defect-inspection\dataset\glcm_features.csv')

df = pd.concat([glcm_df,glh_df],axis=1)
print(df.columns)
X = df.iloc[:,:-1]
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=None)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Principal Component Analysis
pca = PCA().fit(X_train)
Xpca = pca.transform(X_train)
    
# Explained variance ratio by each principal component
v_ratio = pca.explained_variance_ratio_
v_ratio = [v*100 for v in v_ratio]
print('PCA:',v_ratio)

pca = PCA(n_components=3)
X_train, X_test = pca.fit_transform(X_train), pca.transform(X_test)

# SVM Classification
svc = SVC(decision_function_shape='ovo', kernel='rbf',random_state=0)

svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)

print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))