
# coding: utf-8

# In[ ]:

import pandas as pd 
import numpy as np
import lightgbm as lgbm
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn import metrics
import shap 
import sklearn
from sklearn.preprocessing import LabelEncoder

from sklearn import linear_model
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score


# In[ ]:

import lightgbm as lgbm


# In[ ]:

import lightgbm as lgbm


# In[ ]:

columns = []
response_columns= []
#SABLE
#LIMON
#ARGILE


# In[ ]:

#response_columns = ['SABLETG',	'SABLEG',	'SABLEM',	'SABLEF',	'SABLETF',	'SABLE'	,'LIMONF',	'LIMONM',	'LIMONG',	'LIMON'	,'ARGILE',	'BATTANCE'	,'CONDHYD',	'ML',	'MN']
response_columns = ['SABLE', 'LIMON','ARGILE']
columns = ['MO',	'N'	,'INDICE20'	,'PHEAU'	,'PHSMP',	'KECH'	,'CAECH'	,'MGECH',	'NAECH'	,'HECH'	,'CEC',	'PM3',	'MNM3',	'CUM3'	,'FEM3'	,'ALM3'	,'BM3',	'ZNM3',	'PBM3',	'MOM3',	'CDM3',	'COM3',	'CRM3'	,'KM3'	,'CAM3'	,'MGM3',	'NAM3']


# In[ ]:

df = pd.read_excel('Tabi_ML.xlsx', usecols = columns + response_columns)
#chemicals = [x for x in  df.columns if  list(df.columns).index(x) >= list(df.columns).index('MO')]
#columns = df[chemicals]


# In[ ]:

#response_columns = ['SABLETG',	'SABLEG',	'SABLEM',	'SABLEF',	'SABLETF',	'SABLE'	,'LIMONF',	'LIMONM',	'LIMONG',	'LIMON'	,'ARGILE',	'BATTANCE'	,'CONDHYD',	'ML',	'MN']
#df = df[df['TEXTURE'].notna()]
#response_column = df['TEXTURE']

#label_encoder = LabelEncoder()
#label_encoder = label_encoder.fit(response_column)
#label_encoded_y = label_encoder.transform(response_column)
#df['TEXTURE'] = label_encoded_y

#df.drop('TEXTURE', axis = 1, inplace = True)

# Put whatever series you want in its place
#df['TEXTURE'] = label_encoded_y


# In[ ]:

#df['TEXTURE'] = label_encoded_y
#print(df['TEXTURE'].nunique())
print(df.shape)
print(df.head(3))


# In[ ]:


depth = 6
params = {
    'objective'       : 'regression',
    'boosting_type'   : 'gbdt',
    'metric'          : 'rmse',
    'num_leaves'      : 2 ** depth,
    'max_depth'       : depth * 2,
    'learning_rate'   : 0.04,
    'feature_fraction': 0.7,
    'bagging_fraction': 0.75,
    'bagging_seed'    : 42,
    'verbosity'       : 0,
    'random_seed'     : 42,
    'num_threads'     : 4,
    'min_data_in_leaf': 2,

}


# In[ ]:

def rmse(y_true, y_pred):
    return metrics.mean_squared_error(y_true, y_pred) ** 0.5


# In[ ]:

summary = {}

for column in response_columns:
    
    # Keep only rows which have not null value for response variable
    index   = df[column].notnull()
    X_train = df[columns].loc[index]
    y_train = df[column].loc[index]
    
    CV = KFold(n_splits = 5, random_state = 42) 

    fit_score          = [] 
    val_score          = []
    baseline_score     = []
    mean_average_error = []
 
    verbose = False

    for i, (fit_index,val_index) in enumerate(CV.split(X_train, y_train)):

        X_fit = X_train.iloc[fit_index]
        y_fit = y_train.iloc[fit_index]
        X_val = X_train.iloc[val_index]
        y_val = y_train.iloc[val_index]
        
        fit = lgbm.Dataset(X_fit, y_fit)
        val = lgbm.Dataset(X_val, y_val)

        evals_result = {}
        
        # Use mean as baseline model:
        baseline_pred_val = [np.mean(y_val)] * y_val.shape[0]
        baseline_score.append(metrics.mean_absolute_error(y_val, baseline_pred_val))

        model = lgbm.train(
            params,
            fit,
            num_boost_round       = 30000,
            valid_sets            = (fit, val),
            valid_names           = ('fit', 'val'),
            verbose_eval          = 50,
            early_stopping_rounds = 50,
            evals_result          = evals_result,
        )

        pred_fit = model.predict(X_fit)
        pred_val = model.predict(X_val)
        
        mean_average_error.append(metrics.mean_absolute_error(y_val, pred_val))

        if verbose :
            print(f'Rmse fit pour le fold {i+1} : {rmse(pred_fit,Y_fit):.3f}')
            print(f'Rmse val pour le fold fold {i+1} : {rmse(pred_val,Y_val):.3f}')

        fit_score.append(rmse(pred_fit,y_fit))
        val_score.append(rmse(pred_val,y_val))

    fit_score      = np.array(fit_score)
    val_score      = np.array(val_score)
    mae_score      = np.array(mean_average_error)
    baseline_score = np.array(baseline_score)
    
    # Store KPI
    summary[column] = {
        'mae_score'             : np.mean(mae_score), 
        'std_mae_score'         : np.std(mae_score),
        'baseline_score'        : np.mean(baseline_score), 
        'rmse_score'            : np.mean(val_score),
        'std_rmse_score'        : np.std(val_score),
        'model'                 : model,
        'mean_response_variable': np.mean(y_val),
        'n'                     : X_train.shape[0],
        'y_pred'                : pred_val,
        'y_true'                : y_val,
    }

    print(f'RMSE score for fitted data: {np.mean(fit_score):.3f} ± {np.std(fit_score):.3f}')
    print(f'RMSE score for validation data: {np.mean(val_score):.3f} ± {np.std(val_score):.3f}')


# In[ ]:

results_df = pd.DataFrame(summary).T


# In[ ]:

results_df['ratio_mean_mae'] = results_df['mean_response_variable'] / results_df['mae_score']


# In[ ]:

results_df = results_df.sort_values(
    'ratio_mean_mae', 
    ascending = False,
)


# In[ ]:

results_df[['mean_response_variable', 'n', 'rmse_score', 'mae_score', 'std_rmse_score', 'std_mae_score']]


# In[18]:

def explain_model(
    model,
    X_val
):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_val)
    return shap.summary_plot(shap_values, X_val)


# In[19]:

def observed_vs_predicted(
    y_true, 
    y_pred,
):
    plt.scatter(
        y_true,
        y_pred,
        alpha = 0.3,   
    )

    plt.plot(
        y_true, 
        y_true, 
        c = 'red',
    )
    
    plt.xlabel('observed')
    plt.ylabel('predicted')
    return plt.show()


# In[20]:



explain_model(
    model = summary['TEXTURE']['model'],
    X_val = X_val,
)



# In[151]:

'''observed_vs_predicted(
    y_true = summary['TEXTURE']['y_true'], 
    y_pred = summary['TEXTURE']['y_pred'],
)'''


# In[60]:

def xgboost(X, y):
    datamatrix  = xgb.DMatrix(data=X, label=y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,  random_state = 123)
    xg_reg = OneVsRestClassifier(xgb.XGBClassifier())
    params = {"objective": 'reg:squarederror', 'colsample_bytree' : 0.3, "learning_rate" : 0.1 , "max_depth" : 5, "alpha" : 10, "n_estimators" :10}
    #xg_reg = xgb.train(dtrain = Xtrain, dtest = X_test, ytrain= y_train, ytest = y_test, params=params)
    xg_reg.fit(X_train, y_train)
    preds = xg_reg.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    accuracy = accuracy_score(y_test, preds)
    print("accuracy: %f", (accuracy))
    print("RMSE: %f" % (rmse))
    cv_results = xgb.cv(dtrain=datamatrix, params=params, nfold=3,
                    num_boost_round=50,early_stopping_rounds=10,metrics="rmse", as_pandas=True, seed=123)
    print((cv_results["test-rmse-mean"]).tail(1))
    xg_reg = xgb.train(params=params, dtrain=datamatrix, num_boost_round=10)
    #xgb.plot_tree(xg_reg,num_trees=0)
    plt.rcParams['figure.figsize'] = [70, 70]
    plt.rcParams.update({'font.size': 30})
    #plt.savefig('tree.png')
    xgb.plot_importance(xg_reg)
    #plt.rcParams['figure.figsize'] = [5, 5]
    plt.show()


# In[61]:

xgboost(df[columns], df['TEXTURE'])


# In[69]:

from sklearn.cluster import KMeans


# In[146]:

df.dropna(inplace=True)
#indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
X = df[columns].astype('float')
print((X.values.shape)) 
kmeans = KMeans(n_clusters=20, random_state=0).fit(X.values)


# In[150]:

print(kmeans.labels_)


# In[152]:

from sklearn.metrics import silhouette_samples, silhouette_score
preds = kmeans.fit_predict(X.values)
centers = kmeans.cluster_centers_

score = silhouette_score(X.values, preds)
print(score)




'''
depth = 6
params = {
    'objective'       : 'regression',
    'boosting_type'   : 'gbdt',
    'metric'          : 'rmse',
    'num_leaves'      : 2 ** depth,
    'max_depth'       : depth * 2,
    'learning_rate'   : 0.04,
    'feature_fraction': 0.7,
    'bagging_fraction': 0.75,
    'bagging_seed'    : 42,
    'verbosity'       : 0,
    'random_seed'     : 42,
    'num_threads'     : 4,
    'min_data_in_leaf': 2,

}


# In[ ]:

def rmse(y_true, y_pred):
    return metrics.mean_squared_error(y_true, y_pred) ** 0.5


# In[ ]:

summary = {}

for column in response_columns:
    
    # Keep only rows which have not null value for response variable
    index   = df[column].notnull()
    X_train = df[columns].loc[index]
    y_train = df[column].loc[index]
    
    CV = KFold(n_splits = 5, random_state = 42) 

    fit_score          = [] 
    val_score          = []
    baseline_score     = []
    mean_average_error = []
 
    verbose = False

    for i, (fit_index,val_index) in enumerate(CV.split(X_train, y_train)):

        X_fit = X_train.iloc[fit_index]
        y_fit = y_train.iloc[fit_index]
        X_val = X_train.iloc[val_index]
        y_val = y_train.iloc[val_index]
        
        fit = lgbm.Dataset(X_fit, y_fit)
        val = lgbm.Dataset(X_val, y_val)

        evals_result = {}
        
        # Use mean as baseline model:
        baseline_pred_val = [np.mean(y_val)] * y_val.shape[0]
        baseline_score.append(metrics.mean_absolute_error(y_val, baseline_pred_val))

        model = lgbm.train(
            params,
            fit,
            num_boost_round       = 30000,
            valid_sets            = (fit, val),
            valid_names           = ('fit', 'val'),
            verbose_eval          = 50,
            early_stopping_rounds = 50,
            evals_result          = evals_result,
        )

        pred_fit = model.predict(X_fit)
        pred_val = model.predict(X_val)
        
        mean_average_error.append(metrics.mean_absolute_error(y_val, pred_val))

        if verbose :
            print(f'Rmse fit pour le fold {i+1} : {rmse(pred_fit,Y_fit):.3f}')
            print(f'Rmse val pour le fold fold {i+1} : {rmse(pred_val,Y_val):.3f}')

        fit_score.append(rmse(pred_fit,y_fit))
        val_score.append(rmse(pred_val,y_val))

    fit_score      = np.array(fit_score)
    val_score      = np.array(val_score)
    mae_score      = np.array(mean_average_error)
    baseline_score = np.array(baseline_score)
    
    # Store KPI
    summary[column] = {
        'mae_score'             : np.mean(mae_score), 
        'std_mae_score'         : np.std(mae_score),
        'baseline_score'        : np.mean(baseline_score), 
        'rmse_score'            : np.mean(val_score),
        'std_rmse_score'        : np.std(val_score),
        'model'                 : model,
        'mean_response_variable': np.mean(y_val),
        'n'                     : X_train.shape[0],
        'y_pred'                : pred_val,
        'y_true'                : y_val,
    }

    print(f'RMSE score for fitted data: {np.mean(fit_score):.3f} ± {np.std(fit_score):.3f}')
    print(f'RMSE score for validation data: {np.mean(val_score):.3f} ± {np.std(val_score):.3f}')


# In[ ]:

results_df = pd.DataFrame(summary).T


# In[ ]:

results_df['ratio_mean_mae'] = results_df['mean_response_variable'] / results_df['mae_score']


# In[ ]:

results_df = results_df.sort_values(
    'ratio_mean_mae', 
    ascending = False,
)


# In[ ]:

results_df[['mean_response_variable', 'n', 'rmse_score', 'mae_score', 'std_rmse_score', 'std_mae_score']]


# In[18]:

def explain_model(
    model,
    X_val
):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_val)
    return shap.summary_plot(shap_values, X_val)


# In[19]:

def observed_vs_predicted(
    y_true, 
    y_pred,
):
    plt.scatter(
        y_true,
        y_pred,
        alpha = 0.3,   
    )

    plt.plot(
        y_true, 
        y_true, 
        c = 'red',
    )
    
    plt.xlabel('observed')
    plt.ylabel('predicted')
    return plt.show()


# In[20]:



explain_model(
    model = summary['SABLE']['model'],
    X_val = X_val,
)



# In[151]:

observed_vs_predicted(
    y_true = summary['SABLE']['y_true'], 
    y_pred = summary['SABLE']['y_pred'],
)

'''
# In[60]:




