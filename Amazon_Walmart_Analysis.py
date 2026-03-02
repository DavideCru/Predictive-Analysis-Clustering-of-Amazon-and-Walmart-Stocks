import numpy as np 
import pandas as pd 
import yfinance as yf 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.ensemble import RandomForestRegressor 
from sklearn.cluster import KMeans 
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix 
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split 

amzn = yf.download('AMZN', start='2019-01-01', end='2023-12-31') 
wmt = yf.download('WMT', start='2019-01-01', end='2023-12-31') 
 
amzn_price = amzn[['Close']].squeeze() 
wmt_price = wmt[['Close']].squeeze() 

wmt_price = wmt_price.reindex(amzn_price.index) 

data = pd.DataFrame({'AMZN_Close': amzn_price, 'WMT_Close': wmt_price}) 
 
data.interpolate(method='linear', inplace=True) 

for col in data.columns: 
for lag in range(1, 6): 
data[f'{col}_lag{lag}'] = data[col].shift(lag) 

data.dropna(inplace=True) 
 
scaler = StandardScaler() 

data_scaled = scaler.fit_transform(data[['AMZN_Close', 'WMT_Close']]) 
 
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10) 
clusters = kmeans.fit_predict(data_scaled) 
data['Cluster'] = clusters 
 
train = data[data.index < '2023-01-01'] 
test = data[data.index >= '2023-01-01'] 
X_train = train.drop(columns=['AMZN_Close', 'WMT_Close']) 
y_train_amzn = train['AMZN_Close'] 
y_test_amzn = test['AMZN_Close'] 
y_train_wmt = train['WMT_Close'] 
y_test_wmt = test['WMT_Close'] 
X_test = test.drop(columns=['AMZN_Close', 'WMT_Close']) 
 
rf_amzn = RandomForestRegressor(n_estimators=500, random_state=42) 
rf_amzn.fit(X_train, y_train_amzn) 
y_pred_amzn = rf_amzn.predict(X_test) 
rf_wmt = RandomForestRegressor(n_estimators=500, random_state=42) 
rf_wmt.fit(X_train, y_train_wmt) 
y_pred_wmt = rf_wmt.predict(X_test) 
 
rmse_amzn = mean_squared_error(y_test_amzn, y_pred_amzn) ** 0.5 
r2_amzn = r2_score(y_test_amzn, y_pred_amzn) 
rmse_wmt = mean_squared_error(y_test_wmt, y_pred_wmt) ** 0.5 
r2_wmt = r2_score(y_test_wmt, y_pred_wmt) 
print(f"RMSE AMZN: {rmse_amzn:.2f}, R^2 AMZN: {r2_amzn:.2f}") 
print(f"RMSE WMT: {rmse_wmt:.2f}, R^2 WMT: {r2_wmt:.2f}") 

inertia = [] 
K_range = range(1, 10) 
 
for k in K_range: 

kmeans = KMeans(n_clusters=k, random_state=42, n_init=10) 
kmeans.fit(data_scaled) 
inertia.append(kmeans.inertia_) 
plt.figure(figsize=(8,5)) 
plt.plot(K_range, inertia, marker='o') 
plt.xlabel('Numero di Cluster') 
plt.ylabel('Inertia') 
plt.title('Metodo del Gomito') 
plt.show() 
 
conf_matrix = confusion_matrix(data['Cluster'], clusters) 
print('Matrice di confusione:\n', conf_matrix) 
 
corr_matrix = data[['AMZN_Close', 'WMT_Close']].corr() 
plt.figure(figsize=(6,4)) 
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f') 
plt.title('Matrice di correlazione AMZN/WMT') 
plt.show() 

plt.figure(figsize=(12,5)) 
plt.plot(test.index, y_test_amzn, label='AMZN Reale') 
plt.plot(test.index, y_pred_amzn, label='AMZN Predetto', linestyle='dashed') 
plt.legend() 
plt.title('Previsione Random Forest per AMZN') 
plt.show() 
plt.figure(figsize=(12,5)) 
plt.plot(test.index, y_test_wmt, label='WMT Reale') 
plt.plot(test.index, y_pred_wmt, label='WMT Predetto', linestyle='dashed') 
plt.legend() 
plt.title('Previsione Random Forest per WMT') 
plt.show()