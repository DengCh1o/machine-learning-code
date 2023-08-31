import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,r2_score
import matplotlib.pyplot as plt
import numpy as np


df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv')
#####Data preparation as X and Y###
y = df['logS']
x = df.drop('logS',axis = 1)
####Data splitting

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=100)
#####Model building###
####Linear Regressios###

Ir = LinearRegression()
Ir.fit(x_train,y_train)

y_Ir_train_pred = Ir.predict(x_train)
y_Ir_test_pred = Ir.predict(x_test)
print(y_Ir_train_pred,y_Ir_test_pred)

###Evalute model performance

Ir_train_mse = mean_squared_error(y_train,y_Ir_train_pred)
Ir_train_r2 = r2_score(y_train,y_Ir_train_pred)

Ir_test_mse = mean_squared_error(y_test,y_Ir_test_pred)
Ir_test_r2 = r2_score(y_test,y_Ir_test_pred)

print('LR MSE (Train):',Ir_train_mse)
print('Lr R2(Train):',Ir_train_mse)
print('LR MSE (Test):',Ir_test_mse)
print('Lr R2(Test):',Ir_test_mse)
Ir_results = pd.DataFrame(['Linear regreeeion', Ir_train_mse,Ir_train_r2,Ir_test_mse,Ir_test_r2]).transpose()
Ir_results.columns = ['Method','Training MSE','Training R2','Test MSE','Test R2']
#####Random Forest####
####Training the model

rf = RandomForestRegressor(max_depth =2,random_state=100)
rf.fit(x_train,y_train)
RandomForestRegressor(max_depth=2,random_state=100)
#####Applying the model to make a pridiction
y_rf_train_pred = rf.predict(x_train)
y_rf_test_pred = rf.predict(x_test)

rf_train_mse = mean_squared_error(y_train,y_rf_train_pred)
rf_train_r2 = r2_score(y_train,y_rf_train_pred)

rf_test_mse = mean_squared_error(y_test,y_rf_test_pred)
rf_test_r2 = r2_score(y_test,y_rf_test_pred)

rf_results = pd.DataFrame(['Random forest', rf_train_mse,rf_train_r2,rf_test_mse,rf_test_r2]).transpose()
rf_results.columns = ['Method','Training MSE','Training R2','Test MSE','Test R2']


###Model comparison
df_models = pd.concat([Ir_results,rf_results],axis=0)
df_models.reset_index(drop=True) 

###Data visualization of prediction results
plt.figure(figsize=(5,5))
plt.scatter(x=y_train,y=y_Ir_train_pred, c="#7CAE00",alpha= 0.3)

z = np.polyfit(y_train,y_Ir_train_pred,1)
p =np.poly1d(z)

plt.plot(y_train,p(y_train),'#F8766D')
plt.ylabel('Pridict LogS')
plt.xlabel('Experimental LogS')
plt.show()