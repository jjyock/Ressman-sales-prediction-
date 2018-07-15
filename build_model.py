import pandas as pd
import numpy as np
from auto_ml import Predictor
from auto_ml.utils_models import load_ml_model
from sklearn.metrics import mean_squared_error
def rmspe(y_ture,y_pred):
    y_ture=np.array(y_ture)
    y_pred=np.array(y_pred)
    y_ture=y_ture[y_ture>0]
    y_pred = y_pred[y_pred > 0]
    res=((((y_ture-y_pred)/y_ture)**2).sum()/len(y_ture))**0.5
    return res

if __name__=='__main__':
    dtrain=pd.read_csv('data/dtrain.csv')
    dtest = pd.read_csv('data/dtest.csv')
    #print(dtrain.head())
    column_des={
    'DayOfWeek':'categorical',
    'StateHoliday':'categorical',
    'month':'categorical',
    'day':'categorical',
    'has_promot':'categorical',
    'StoreType':'categorical',
    'Assortment':'categorical',
    'Sales':'output',
    'Store':'ignore',
    'Date':'ignore',
    'Customers':'ignore',
    'year':'ignore'}
    model_names = ['LGBMRegressor', 'XGBRegressor']
    #ml_predictor = Predictor(type_of_estimator='regressor', column_descriptions=column_des)
    #ml_predictor.train(dtrain,model_names=model_names,optimize_final_model=True)
    #ml_predictor.save()
    ml_predictor=load_ml_model('auto_ml_saved_pipeline_gb.dill')
    res=ml_predictor.predict(dtest)
    print(rmspe(dtest.Sales,res))

