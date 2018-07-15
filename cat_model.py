from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from catboost import Pool, CatBoostClassifier, cv
import pandas as pd
import numpy as np
import hyperopt
from itertools import combinations
def rmspe(y_ture,y_pred):
    y_ture=np.array(y_ture)
    y_pred=np.array(y_pred)
    res=((((y_ture-y_pred)/y_ture)**2).sum()/len(y_ture))**0.5
    return res


class catboost_model():
    def __init__(self,X_train,X_test,y_train,y_test,cat_index):
        self.X_train=X_train
        self.X_test=X_test
        self.y_train=y_train
        self.y_test=y_test
        self.cat_index=cat_index
    def parameter_search(self):
        def hyperopt_objective(params):
            X_train=self.X_train
            y_train=self.y_train
            categorical_features_indices=self.cat_index
            model = CatBoostRegressor(
                l2_leaf_reg=int(params['l2_leaf_reg']),
                learning_rate=params['learning_rate'],
                depth=params['tree_depth'],
                #iterations=500,
                eval_metric='RMSE',
                #use_best_model=True,
                random_seed=42,
                logging_level='Silent'
            )

            cv_data = cv(
                params=model.get_params(),
                pool=Pool(X_train, y_train, cat_features=categorical_features_indices)
            )
            #print(cv_data)
            best_rmse = np.min(cv_data['test-RMSE-mean'])
            print('params is', params,'rmse is ',best_rmse)
            return best_rmse  # as hyperopt minimises
        params_space = {
            'l2_leaf_reg': hyperopt.hp.qloguniform('l2_leaf_reg', 0, 2, 1),
            'learning_rate': hyperopt.hp.uniform('learning_rate', 1e-3, 5e-1),
            'tree_depth':hyperopt.hp.randint('tree_depth',4)+6

        }

        trials = hyperopt.Trials()

        best = hyperopt.fmin(
            hyperopt_objective,
            space=params_space,
            algo=hyperopt.tpe.suggest,
            max_evals=20,
            trials=trials
        )
        best['tree_depth']=best['tree_depth']+6
        print('best param is ',best)
        return best
    def cat_predict(self,X_pred,search=True):
        if search:
            param=self.parameter_search()
            print(param)
            model = CatBoostRegressor(l2_leaf_reg=int(param['l2_leaf_reg']),
                                      learning_rate=param['learning_rate'],
                                      depth=param['tree_depth'],
                                      loss_function='RMSE').fit(self.X_train, self.y_train, cat_features=self.cat_index,
                                                            use_best_model=False, eval_set=(self.X_test,self.y_test))
        else:
            model=CatBoostRegressor(loss_function='RMSE').fit(X_train, y_train, cat_features=self.cat_index,
                                                            use_best_model=False, eval_set=(self.X_test,self.y_test))
        preds=model.predict(X_pred)
        print(rmspe(self.y_test,preds))

def get_feature_index(data,cat_feature_list):
    cat_feature = []
    for feature in cat_feature_list:
        cat_feature.extend(np.where(data.columns == feature)[0].tolist())
    return cat_feature
def get_data():
    data=pd.read_csv('data/split/data_no.csv')
#print(data.head())
    data_no_X=data[['Store','DayOfWeek','Open','Promo','StateHoliday','SchoolHoliday','month','day',
             'StoreType','Assortment','CompetitionDistance']]
    data_no_y=data['Sales']
    cat_feature=[0,1,2,3,4,5,6,7,8,9]
    cat_list=['Store','DayOfWeek','Open','Promo','StateHoliday','SchoolHoliday','month','day',
             'StoreType','Assortment']
    cat_feature_index=get_feature_index(data_no_X,cat_list)
    print(cat_feature_index)
    X_train,X_test,y_train,y_test=train_test_split(data_no_X,data_no_y)
    return X_train,X_test,y_train,y_test,cat_feature_index

#model=CatBoostRegressor(loss_function='RMSE').fit(X_train,y_train,cat_features=cat_feature,use_best_model=False,
#                                                  eval_set=(X_test,y_test))
#y_pred=model.predict(X_test)
#print('rmsep',rmspe(y_test,y_pred))
if __name__=='__main__':
    X_train, X_test, y_train, y_test, cat_feature_index=get_data()
    catboost_model(X_train,X_test,y_train,y_test,cat_feature_index).cat_predict(X_test)
