import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler,Imputer
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.decomposition import PCA
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# from sklearn.model_selection import KFold
# from sklearn.model_selection import cross_val_score
# import scipy

###Preprocess data
# df0 = pd.read_excel(open("Task_Data_Scientist_Dataset.xlsx",'rb'),sheetname = 0)
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

df.head(3)
#handle missing value, I dont need, now all the "Nan "is missing
# fill missing values with median column values
# X = df.loc[:, df.columns != 'Sex'].values
imputer = Imputer(missing_values='NaN', strategy='median')
transformed_values = imputer.fit_transform(df)

###########Separate data to prediction and training data
mask = df['Sale_MF'] == 1
# mask_p = df['Sale_MF'] == 0
X_training = df[mask]
X_prediction = df[~mask]

# evaluate an LDA model on the dataset using k-fold cross validation
# model = LinearDiscriminantAnalysis()
# kfold = KFold(n_splits=3, random_state=7)
# result = cross_val_score(model, transformed_values, MF, cv=kfold, scoring='accuracy')
# print(result.mean())

## rescale data
#M=1, Female = 0
#other data with Rescale Data(0,1)
#rescaler
# array = transformed_values.values
New_X_training = X_training.values[:,1:-6]
New_Y_training = X_training.values[:,-3]
scaler = MinMaxScaler(feature_range=(0, 1))
rescaledX = scaler.fit_transform(New_X_training) #no Id and SEX
####training/test set raio=0.6/0.4
X_train, X_test, y_train, y_test = train_test_split( rescaledX, New_Y_training, test_size=0.4, random_state=4)
#append column
#put header
# rescaledX = pd.DataFrame(rescaledX, columns=df.columns.values[df.columns != 'Sex'])
####Decide important feature
# model = DecisionTreeRegressor()
# model.fit(rescaledX, Y)
# fit_ratio=0
# i = 0
# while fit_ratio< 0.8 :
#     i+=1
#     pca = PCA(n_components=i)
#     fit = pca.fit(X_train, y_train)
#     fit_ratio = sum(fit.explained_variance_ratio_)
#     print "number of component: " , i
#     print "Explained Variance: " , sum(fit.explained_variance_ratio_)

from sklearn import preprocessing
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

##print the training score
y_pred = model.predict(X_test)
y_test2 = lab_enc.fit_transform(y_test)
print(metrics.accuracy_score(y_test2, y_pred))
#prediction

New_X_prediction = X_prediction.values[:,1:-6]
New_Y_prediction = X_prediction.values[:,-3]
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











# try K=1 through K=25 and record testing accuracy
k_range = range(1, 25)
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    y_train2 = lab_enc.fit_transform(y_train)
    knn.fit(X_train, y_train2)
    y_pred = knn.predict(X_test)
    scores.append(metrics.accuracy_score(y_test2, y_pred))

print(scores)

#try kNeighbor
knn = KNeighborsClassifier(n_neighbors=1)
y_train2 = lab_enc.fit_transform(y_train)
knn.fit(X_train, y_train2)
y_pred = knn.predict(X_train)
print(metrics.accuracy_score(y_train2, y_pred))