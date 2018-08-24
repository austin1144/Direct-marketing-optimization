import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler,Imputer
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from sklearn import preprocessing
#parameter


###Preprocess data
df1 = pd.read_excel(open("Task_Data_Scientist_Dataset.xlsx",'rb'),sheetname = 1)
df2 = pd.read_excel(open("Task_Data_Scientist_Dataset.xlsx",'rb'),sheetname = 2)
df3 = pd.read_excel(open("Task_Data_Scientist_Dataset.xlsx",'rb'),sheetname = 3)
df4 = pd.read_excel(open("Task_Data_Scientist_Dataset.xlsx",'rb'),sheetname = 4)
# mark NaN values as 0, it's actually 0
df1 = df1.replace("F",0)
df1 = df1.replace("M",1)
df2 = df2.replace(np.NaN, 0 )
df3 = df3.replace(np.NaN, 0)
df4 = df4.replace(np.NaN, 0 )
df = pd.merge(left=df1,right=df2, how='left', left_on='Client',right_on='Client')
df = pd.merge(left=df,right=df3, how='left', left_on='Client',right_on='Client')
df = pd.merge(left=df,right=df4, how='left', left_on='Client',right_on='Client')
#replace revenue Nan as 0
for i in df4.columns.values:
    df[i] = df[i].replace(np.NaN,0)
#handle missing value, I dont need, now all the "Nan "is missing
# fill missing values with median column values
imputer = Imputer(missing_values='NaN', strategy='median')
transformed_values = imputer.fit_transform(df)

###########Separate data to prediction and training data
mask = df['Sale_CC'] == 1
# mask_p = df['Sale_CC'] == 0
X_training = df[mask]
X_prediction = df[~mask]

New_X_training = X_training.values[:,1:-6]
New_Y_training = X_training.values[:,-2]
scaler = MinMaxScaler(feature_range=(0, 1))
New_X_training = imputer.fit_transform(New_X_training)
rescaledX = scaler.fit_transform(New_X_training) #no Id and SEX
####training/test set raio=0.6/0.4
X_train, X_test, y_train, y_test = train_test_split( rescaledX, New_Y_training, test_size=0.4, random_state=4)


lab_enc = preprocessing.LabelEncoder()
encode_Y = lab_enc.fit_transform(y_train)
model = ExtraTreesClassifier()
model.fit(X_train, encode_Y )

###print importance
XX = X_training.columns.values[1:-6]
YY = model.feature_importances_
plt.plot(YY, 'r--')
for i in range(len(XX)):
    print (XX[i],YY[i])

New_X_prediction = X_prediction.values[:,1:-6]
New_Y_prediction = X_prediction.values[:,-2]
imputer = Imputer(missing_values='NaN', strategy='median')
New_X_prediction = imputer.fit_transform(New_X_prediction)
scaler = MinMaxScaler(feature_range=(0, 1))
rescaledX = scaler.fit_transform(New_X_prediction)

y_pred = model.predict(rescaledX)
XX = X_prediction.values[:,0]
YY = y_pred
l = len(XX)
for i in range(l):
    print (XX[i],YY[i])