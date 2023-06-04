import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

glh_df = pd.read_csv(r'C:\Users\LENOVO\CV\textile-defect-inspection\dataset\glh_features.csv')
glcm_df = pd.read_csv(r'C:\Users\LENOVO\CV\textile-defect-inspection\dataset\glcm_features.csv')

df = pd.concat([glcm_df,glh_df],axis=1)

X = df.iloc[:,:-1]
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=None)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

xgb_classifier = xgb.XGBClassifier()
xgb_classifier.fit(X_train,y_train)

y_pred = xgb_classifier.predict(X_test)

print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))