import pandas as pd
import numpy as np
import random
import math
import time
import xgboost as xgb
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
start=datetime.now()
print(" -> Reading data files... ",start)
data_dir = ''
train = pd.read_csv('train.csv' )
test = pd.read_csv('test.csv' )
store = pd.read_csv('store.csv' )
store.fillna(-1, inplace=True)
print(" -> Remove columns with Sales = 0, Open = 0")
train = train[(train['Open']==1)&(train['Sales']>0)]
print(" -> Join with Store table")
train = train.merge(store, on = 'Store', how = 'left')
test = test.merge(store, on = 'Store', how = 'left')
print(" -> Process the Date column")
for ds in [train, test]:
    tmpDate = [time.strptime(x, '%Y-%m-%d') for x in ds.Date]
    ds['mday'] = [x.tm_mday for x in tmpDate]
    ds['mon'] = [x.tm_mon for x in tmpDate]
    ds['year'] = [x.tm_year for x in tmpDate]
print(" -> Process categorical columns")
for f in ['StateHoliday', 'StoreType', 'Assortment', 'PromoInterval']:
    clf =LabelEncoder().fit(train[f].astype(str))
    for ds in [train, test]:
        ds[f] = clf.transform(ds[f].astype(str))
def generate_feature(df):
    df['Sales']=df['Sales'].astype(float)
    df['Customers']=df['Customers'].astype(float)
    store_data_sales = df.groupby('Store')['Sales'].sum()
    store_data_customers = df.groupby('Store')['Customers'].sum()
    store_data_open = df.groupby('Store')['Open'].count()
    store_data_sales_per_day = store_data_sales / store_data_open
    store_data_customers_per_day = store_data_customers / store_data_open
    store_data_sales_per_customer_per_day = store_data_sales_per_day / store_data_customers_per_day
    return pd.concat([store_data_sales_per_day,store_data_customers_per_day,store_data_sales_per_customer_per_day],axis=1)

data_new_feature=generate_feature(train)
train=train.merge(data_new_feature,left_on='Store',right_index=True)
test=test.merge(data_new_feature,left_on='Store',right_index=True)
data_new_feature.columns=['store_data_sales_per_day','store_data_customers_per_day','store_data_sales_per_customer_per_day']
store_features = ['StoreType', 'Assortment', 'CompetitionDistance', 'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 'Promo2', 'Promo2SinceWeek', 'Promo2SinceYear', 'PromoInterval']
features = ['Store','DayOfWeek', 'Promo', 'StateHoliday', 'SchoolHoliday', 'mday', 'mon', 'year'] + store_features
print(" -> XGBoost Train")
h = random.sample(range(len(train)),10000)
dvalid = xgb.DMatrix(train.ix[h][features].values, label=[math.log(x+1) for x in train.ix[h]['Sales'].values])
dtrain = xgb.DMatrix(train.drop(h)[features].values, label=[math.log(x+1) for x in train.drop(h)['Sales'].values])
param = {'objective': 'reg:linear',
    'eta': 0.01,
        'booster' : 'gbtree',
            'max_depth':12,
                'subsample':0.8,
                    'silent' : 1,
                        #'nthread':6,
                        #'tree_method':'gpu_hist',
                        #'n_gpus':2,
                        'colsample_bytree':0.7}
def evalerror(preds, dtrain):
    labels = dtrain.get_label()
    # return a pair metric_name, result
    y = [math.exp(x)-1 for x in labels[labels > 0]]
    yhat = [math.exp(x)-1 for x in preds[labels > 0]]
    ssquare = [math.pow((y[i] - yhat[i])/y[i],2) for i in range(len(y))]
    return 'rmpse', math.sqrt(np.mean(ssquare))
watchlist = [(dtrain,'train_rmpse'),(dvalid,'valid_rmpse')]
clf = xgb.train(param, dtrain, 10000, watchlist,feval=evalerror,verbose_eval=100)
dtest = xgb.DMatrix(test[test['Open']==1][features].values)
test['Sales'] = 0
test.loc[test['Open']==1,'Sales'] = [math.exp(x) - 1 for x in clf.predict(dtest)]
print("-> Write submission file ... ")
print(datetime.now(),datetime.now()-start)
test[['Id', 'Sales']].to_csv("submission.csv", index = False)
