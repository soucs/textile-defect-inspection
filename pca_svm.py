import pandas as pd
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

df = pd.read_csv(r'/home/soucs/Python/textile-defect-inspection/dataset/textile_defects.csv')
df['indication_value'] = df['indication_value'].apply(lambda label:1 if label!=0 else 0)

X = df[['c1', 'c2', 'c3', 'c4', 'c5']]
y = df['indication_value']


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.5, random_state=1)
print((y_test==0).sum())


pca = PCA()
pca.fit(X_train)
Xpca = pca.transform(X_train)
    
# Explained variance ration by each principal component
v_ratio = pca.explained_variance_ratio_
v_ratio = [round(v*100,2) for v in v_ratio]

print(v_ratio)

# Taking components that explain atleast 95% of variance
count, explained_v = 0, 0
for v in v_ratio:
    explained_v += v
    count += 1
    if explained_v>=95:
        break

# Getting only significant principal components
signif_pca = PCA(n_components=count).fit(X_train)

# Applying dimmensionality reduction
X_train_final = signif_pca.transform(X_train)
X_test_final = signif_pca.transform(X_test)

print(X_train_final)


# svc = SVC(decision_function_shape='ovo')
# svc = SVC()

# svc.fit(X_train, y_train)
# y_pred = svc.predict(X_test)

# print(accuracy_score(y_test, y_pred)) # 0.36920833333333336 for non-binary
# print(precision_score(y_test, y_pred))
# print(recall_score(y_test, y_pred))
# print(y_test[y_test!=y_pred].sum())

'''Giving 83% accuracy for defect/non-defect binary classification'''

