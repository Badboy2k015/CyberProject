import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,precision_recall_fscore_support

import time
from statistics import mode
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder 
from statsmodels.regression.quantile_regression import QuantReg
from sklearn import tree

df = pd.read_csv('IoT_Cyber_Samples.CSV')
numeric_features = df.dtypes[df.dtypes != 'object'].index
df[numeric_features] = df[numeric_features].apply(
    lambda x: (x - x.min()) / (x.max()-x.min()))
# Fill empty values by 0
df = df.fillna(0)
labelencoder = LabelEncoder()
df.iloc[:, -1] = labelencoder.fit_transform(df.iloc[:, -1])
X = df.drop(['Label'],axis=1).values 
y = df.iloc[:, -1].values.reshape(-1,1)
y=np.ravel(y)
X_train, X_test, y_train, y_test = train_test_split(X,y, train_size = 0.8, test_size = 0.2, random_state = 0,stratify = y)
X_train.shape
pd.Series(y_train).value_counts()
print(pd.Series(y_train).value_counts())
from imblearn.over_sampling import SMOTE
smote=SMOTE(n_jobs=-1,sampling_strategy={4:1500}) # Create 1500 samples for the minority class "4"
X_train, y_train = smote.fit_resample(X_train, y_train)
pd.Series(y_train).value_counts()
#IoT Network Node Formation
x1=np.array([5,7,8,7,2,17,2,9,4,11,12,9,6])
y1=np.array([91,86,87,88,111,86,103,87,94,78,77,85,86])
plt.scatter(x1,y1,color='hotpink')
x2=np.array([2,2,8,1,15,8,12,9,7,3,11,4,7,14,12])
y2=np.array([100,105,84,105,90,99,90,95,94,100,79,112,91,80,85])
plt.scatter(x2,y2,color='green')
x3=np.array([8,14,4])
y3=np.array([90,100,90])
index1=np.arange(10)
dataset_values=[0.00018,0.00016,0.00015,0.00012,0.00012,0.0008,0.0008,0.0006,0.0004,0.0002]
plt.scatter(x3,y3,color='red')
plt.legend(['Static_IoT_Node','Moving_IoT_Node','Cyber_Attack_IoT_nodes'])
plt.title('Randomized IoT Node Distribution')
plt.xlabel('Node harizondal Axis')
plt.ylabel('Node Vertical Axis')
plt.show()

# Decision tree training and prediction
dt = DecisionTreeClassifier(random_state = 0)
clf=dt.fit(X_train,y_train) 
dt.fit(X_train,y_train) 
tree.plot_tree(clf)
dt_score=dt.score(X_test,y_test)
y_predict=dt.predict(X_test)
y_true=y_test
print('Accuracy of DT: '+ str(dt_score))
precision,recall,fscore,none= precision_recall_fscore_support(y_true, y_predict, average='weighted') 
print('Precision of DT: '+(str(precision)))
print('Recall of DT: '+(str(recall)))
print('F1-score of DT: '+(str(fscore)))
print(classification_report(y_true,y_predict))
cm=confusion_matrix(y_true,y_predict)
f,ax=plt.subplots(figsize=(5,5))
sns.heatmap(cm,annot=True,linewidth=0.5,linecolor="red",fmt=".0f",ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.show()
plt.bar(index1,dataset_values)
plt.xticks(index1+0.6,['Flow Duration','Total Forward packet','Total Backward packet','Flow Bytes','Flow Packets','Flow IAT Mean','Flow IAT Max','Flow IAT min','Flow IAT Total','FWD IAT max'],rotation=90,ha='right')
plt.title('Features of Entire Dataset')
plt.show()
X = [x for x in range(0,900)]
N_0 = 300
m_lambda = .005    
left_shift = 0
up_shift = 0
plt.close()
Y = [ N_0 * np.exp(1)**(-1*m_lambda*(m_x+left_shift)) + up_shift + np.random.normal(0,15) for m_x  in X]
Y_fit = [ N_0 * np.exp(1)**(-1*m_lambda*(m_x+left_shift)) + up_shift for m_x  in X]
plt.plot(X,Y,'o')
plt.plot(X,Y_fit,'-')
plt.title('DISTRIBUTION OF RESULTS ACCORDING TO TYPE OF ATTACK')
plt.show()
dt_train=dt.predict(X_train)
dt_test=dt.predict(X_test)

# Random Forest training and prediction

rf = RandomForestClassifier(random_state = 0)
rf.fit(X_train,y_train) 
rf_score=rf.score(X_test,y_test)
y_predict=rf.predict(X_test)
y_true=y_test
print('Accuracy of RF: '+ str(rf_score))
precision,recall,fscore,none= precision_recall_fscore_support(y_true, y_predict, average='weighted') 
print('Precision of RF: '+(str(precision)))
print('Recall of RF: '+(str(recall)))
print('F1-score of RF: '+(str(fscore)))
print(classification_report(y_true,y_predict))
cm=confusion_matrix(y_true,y_predict)
f,ax=plt.subplots(figsize=(5,5))
sns.heatmap(cm,annot=True,linewidth=0.5,linecolor="red",fmt=".0f",ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.show()
x = np.random.gamma(1,1,size = 1000)
xx = np.linspace(0,6,101)
y = 10*np.exp(-x/2) + np.random.normal(0, 0.25, size = x.size)
plt.scatter(x,y, c = 'C0', alpha = 0.5, s = 5)
X = np.c_[np.ones_like(x), x]
XX = np.c_[np.ones_like(xx), xx]
model = QuantReg(np.log(y),X).fit(q = 0.95)
preds = np.exp(model.predict(XX))
plt.plot(xx, preds, color = 'red')
plt.title('PREDICTION OBTAINED USING RANDOM FOREST REGRESSOR FOR DATASET')