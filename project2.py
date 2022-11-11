import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
train_data=pd.read_csv('train.csv')
test_data=pd.read_csv('test.csv')
print(train_data.shape)
print(test_data.shape)
test2=np.array(test_data['PassengerId'],dtype=str)
print(test2)
col=train_data.shape[1]
x_train=train_data.iloc[:,1:col-1]
y_train=train_data.iloc[:,col-1:col]
print(x_train.head())
print(y_train.head())
list=['HomePlanet','CryoSleep','Cabin','Destination','VIP','Name']
labelencoder=LabelEncoder()
for li in list:
    x_train[li]=labelencoder.fit_transform(x_train[li])
    test_data[li]=labelencoder.fit_transform(test_data[li])
y_train['Transported']=labelencoder.fit_transform(y_train['Transported'])    
impute=SimpleImputer(missing_values=np.nan,strategy='mean')
x_train=impute.fit_transform(x_train)
test_data=impute.fit_transform(test_data)
x_train2,x_test,y_train2,y_test=train_test_split(x_train,y_train,test_size=.20,shuffle=True,random_state=33)
print(x_train2.shape)
print(y_train2.shape)
print(x_test.shape)
print(y_test.shape)
randomclassifier=RandomForestClassifier(max_depth=20)
randomclassifier.fit(x_train2,y_train2)
print("RandomForestClassifier train score = ",randomclassifier.score(x_train2,y_train2))
print("RandomForestClassifier test score = ",randomclassifier.score(x_test,y_test))
y_pred=randomclassifier.predict(x_test)
confusionmatrix=confusion_matrix(y_test, y_pred)
sns.heatmap(confusionmatrix,center=True)
test=randomclassifier.predict(test_data[:,1:])
print(test[:10])
submission=[]
for x in test:
      if x==0:
          submission.append('FALSE')
      else:
          submission.append('TRUE')
print(submission)
submission2=pd.DataFrame({'PassengerId':test2,'Transported':submission})   
print(submission2)   
submission2.to_csv('classfication_test')    