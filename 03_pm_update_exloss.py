import random
import os
import pprint
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, StratifiedGroupKFold,GroupKFold, GroupShuffleSplit, ShuffleSplit
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score, StratifiedKFold, learning_curve
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from lightgbm import LGBMRegressor
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from xgboost import XGBRegressor
import xgboost as xgb
from typing import Tuple
from shapely.geometry import Point
from sklearn.model_selection import train_test_split, StratifiedGroupKFold,GroupKFold, GroupShuffleSplit, ShuffleSplit, KFold
from sklearn.metrics import mean_squared_error
import math
from sklearn.preprocessing import LabelBinarizer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import KNNImputer
from collections import Counter
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from shapely.geometry import Point
import unittest
from sklearn.model_selection import train_test_split, StratifiedGroupKFold,GroupKFold, GroupShuffleSplit, ShuffleSplit, KFold
import random
import geopandas as gpd
import shapely
from shapely.geometry.polygon import Polygon
from collections import Counter
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, StratifiedGroupKFold,GroupKFold, GroupShuffleSplit, ShuffleSplit
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score, StratifiedKFold, learning_curve
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score
from sklearn.preprocessing import Normalizer
import sys
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
from sklearn.metrics import r2_score 
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle
import random 
import itertools
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error

# Custom loss function which uses Exloss internal logic for XGBoost
def custom_loss(preds, dtrain, up_th=0.9, lamda_underestimate=1.2, lamda_overestimate=1.0, lamda=1.0):
    
    labels = dtrain.get_label()
    
    # MSE components
    mse_grad = 2 * (preds - labels)
    mse_hess = 2 * np.ones_like(preds)

    # Quantile threshold for the upper range
    tar_up = np.quantile(labels, q=up_th)
    
    # Calculate target and prediction areas for the upper quantile
    target_up_area = np.maximum(labels - tar_up, 0)
    pred_up_area = np.maximum(preds - tar_up, 0)
    
    # Loss adjustments for overestimations and underestimations
    loss_up_over = lamda_overestimate * np.maximum(pred_up_area - target_up_area, 0)
    loss_up_under = lamda_underestimate * np.maximum(target_up_area - pred_up_area, 0)
    
    grad = mse_grad + (lamda / (1 - up_th)) * (
        -2 * loss_up_under * (target_up_area - pred_up_area) +
         2 * loss_up_over * (pred_up_area - target_up_area))
    
    hess = mse_hess + (lamda / (1 - up_th)) * (
         2 * (loss_up_under + loss_up_over))
         
    return grad, hess

record = 'exloss'
record2 = 'mse'

path = "/scratch/users/akawano/pm_update/data/intermediate/"  # Dataset location
pm25 = pd.read_csv(os.path.join(path, "df_ml.csv"))
grid_india_monitor_region = gpd.read_file(os.path.join(path, "grid_india_monitor_region"))
grid_india_monitor_region = grid_india_monitor_region[['grid_id', 'grid_id_50km', 'k_region', 'geometry']].copy()

pm25['grid_id'] = pm25['grid_id'].astype(int).astype(str)
grid_india_monitor_region['grid_id'] = grid_india_monitor_region['grid_id'].astype(int).astype(str)
pm25 = pm25.merge(grid_india_monitor_region, how='left', on='grid_id')

train_k_region1 = pm25.loc[pm25['k_region'] == 1]
train_k_region1 = train_k_region1.reset_index(drop = True)
train_k_region1_copy = train_k_region1.copy() 
y = train_k_region1_copy.pop('pm25').to_frame()
X = train_k_region1_copy
gkf = GroupKFold(n_splits=10) 
train_dfs_region1 = []
test_dfs_region1 = []
for train, test in gkf.split(X, y, groups=X['grid_id_50km']):
    train = train_k_region1.loc[train]
    train_dfs_region1.append(train)
    test = train_k_region1.loc[test]
    test_dfs_region1.append(test)
    
train_k_region2 = pm25.loc[pm25['k_region'] == 2]
train_k_region2 = train_k_region2.reset_index(drop = True)
train_k_region2_copy = train_k_region2.copy() 
y = train_k_region2_copy.pop('pm25').to_frame()
X = train_k_region2_copy
gkf = GroupKFold(n_splits=10) 
train_dfs_region2 = []
test_dfs_region2 = []
for train, test in gkf.split(X, y, groups=X['grid_id_50km']):
    train = train_k_region2.loc[train]
    train_dfs_region2.append(train)
    test = train_k_region2.loc[test]
    test_dfs_region2.append(test)

train_k_region3 = pm25.loc[pm25['k_region'] == 3]
train_k_region3 = train_k_region3.reset_index(drop = True)
train_k_region3_copy = train_k_region3.copy() 
y = train_k_region3_copy.pop('pm25').to_frame()
X = train_k_region3_copy
gkf = GroupKFold(n_splits=10) 
train_dfs_region3 = []
test_dfs_region3 = []
for train, test in gkf.split(X, y, groups=X['grid_id_50km']):
    train = train_k_region3.loc[train]
    train_dfs_region3.append(train)
    test = train_k_region3.loc[test]
    test_dfs_region3.append(test)
    

# Create an empty list to store the concatenated dataframes
train_concat = []

# Iterate over the indices of the lists
for i in range(10):
    concatenated_df = pd.concat([train_dfs_region1[i], train_dfs_region2[i], train_dfs_region3[i]], axis=0, ignore_index=True)
    concatenated_df = shuffle(concatenated_df).reset_index(drop = True)
    train_concat.append(concatenated_df)

test_concat = []

# Iterate over the indices of the lists
for i in range(10):
    concatenated_df = pd.concat([test_dfs_region1[i], test_dfs_region2[i], test_dfs_region3[i]], axis=0, ignore_index=True)
    concatenated_df = shuffle(concatenated_df).reset_index(drop = True)
    test_concat.append(concatenated_df)

### inner CV for hyperparameters ####

train = train_concat[0]

train_k_region1 = train.loc[train['k_region'] == 1]
train_k_region1['index'] = train_k_region1.index
train_k_region1_copy = train_k_region1.copy()
y = train_k_region1_copy.pop('pm25').to_frame()
X = train_k_region1_copy
gkf = GroupKFold(n_splits=5) 
train_indices_region1 = []
test_indices_region1 = []
for train_idx, test_idx in gkf.split(X, y, groups=X['grid_id_50km']):
    train_index = train_k_region1.iloc[train_idx]
    train_index = train_index['index'].values
    test_index = train_k_region1.iloc[test_idx]
    test_index = test_index['index'].values
    
    train_indices_region1.append(train_index)
    test_indices_region1.append(test_index)
    
train_k_region2 = train.loc[train['k_region'] == 2]
train_k_region2['index'] = train_k_region2.index
train_k_region2_copy = train_k_region2.copy()
y = train_k_region2_copy.pop('pm25').to_frame()
X = train_k_region2_copy
gkf = GroupKFold(n_splits=5) 
train_indices_region2 = []
test_indices_region2 = []
for train_idx, test_idx in gkf.split(X, y, groups=X['grid_id_50km']):
    train_index = train_k_region2.iloc[train_idx]
    train_index = train_index['index'].values
    test_index = train_k_region2.iloc[test_idx]
    test_index = test_index['index'].values
    
    train_indices_region2.append(train_index)
    test_indices_region2.append(test_index)
    
train_k_region3 = train.loc[train['k_region'] == 3]
train_k_region3['index'] = train_k_region3.index
train_k_region3_copy = train_k_region3.copy()
y = train_k_region3_copy.pop('pm25').to_frame()
X = train_k_region3_copy
gkf = GroupKFold(n_splits=5) 
train_indices_region3 = []
test_indices_region3 = []
for train_idx, test_idx in gkf.split(X, y, groups=X['grid_id_50km']):
    train_index = train_k_region3.iloc[train_idx]
    train_index = train_index['index'].values
    test_index = train_k_region3.iloc[test_idx]
    test_index = test_index['index'].values
    
    train_indices_region3.append(train_index)
    test_indices_region3.append(test_index)

train_indices= []
# Iterate over the indices of the lists
for i in range(5):
    trn_indices = [*train_indices_region1[i], *train_indices_region2[i], *train_indices_region3[i]]
    train_indices.append(random.sample(trn_indices, len(trn_indices))) 
    
    
test_indices= []
# Iterate over the indices of the lists
for i in range(5):
    te_indices = [*test_indices_region1[i], *test_indices_region2[i], *test_indices_region3[i]]
    test_indices.append(random.sample(te_indices, len(te_indices))) 

city_cv = [*zip(train_indices, test_indices)]
train_nested = train.drop(columns = ['date', 'grid_id', 'grid_id_50km', 'k_region', 'geometry'])
y_train_nested = train_nested.pop("pm25").to_frame() 
X_train_nested = train_nested

# # # # # # # # # # # # # # # # # # # # # # 
params_XGB_exloss = {
    'max_depth': [10],
    'learning_rate': [0.1],
    #'n_estimators': [1500],
    #'num_leaves': [1023],
    'max_bin': [1000],
    'colsample_bytree': [0.8],
    'min_child_weight': [1],
    'reg_lambda': [50, 100, 1000],
    #'tree_method': ['gpu_hist']
    #'objective': ['reg:squarederror'],
    'tree_method': ['hist']  # or 'exact', as applicable
}

#{'max_depth': 10, 'learning_rate': 0.1, 'max_bin': 1000, 'colsample_bytree': 1, 'min_child_weight': 1, 'reg_lambda': 10, 'tree_method': 'hist'}

# Create all combinations of parameters
keys, values = zip(*params_XGB_exloss.items())
param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

scoring = {'r_squared': 'r2', 'rmse': 'neg_root_mean_squared_error', 'mape': 'neg_mean_absolute_percentage_error'}
best_score = float('inf')
best_params_XGB_exloss = None
best_cv_results = {}

# The number of splits for your cross-validation
num_splits = 5

for params in param_combinations:
    fold_results = {
        'train_r_squared': [],
        'test_r_squared': [],
        'train_rmse': [],
        'test_rmse': [],
        'train_mape': [],
        'test_mape': [],
    }
    
    for train_idx, test_idx in city_cv:
        dtrain = xgb.DMatrix(X_train_nested.iloc[train_idx], label=y_train_nested.iloc[train_idx].values.ravel())
        dtest = xgb.DMatrix(X_train_nested.iloc[test_idx], label=y_train_nested.iloc[test_idx].values.ravel())
        
        bst = xgb.train(params, dtrain, num_boost_round = 2000, obj=custom_loss)
        
        train_pred = bst.predict(dtrain)
        test_pred = bst.predict(dtest)
        
        # Calculate metrics for train and test datasets
        train_r2 = r2_score(y_train_nested.iloc[train_idx].values.ravel(), train_pred)
        test_r2 = r2_score(y_train_nested.iloc[test_idx].values.ravel(), test_pred)
        
        train_rmse = np.sqrt(mean_squared_error(y_train_nested.iloc[train_idx].values.ravel(), train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_train_nested.iloc[test_idx].values.ravel(), test_pred))
        
        train_mape = mean_absolute_percentage_error(y_train_nested.iloc[train_idx].values.ravel(), train_pred)
        test_mape = mean_absolute_percentage_error(y_train_nested.iloc[test_idx].values.ravel(), test_pred)
        
        fold_results['train_r_squared'].append(train_r2)
        fold_results['test_r_squared'].append(test_r2)
        fold_results['train_rmse'].append(train_rmse)
        fold_results['test_rmse'].append(test_rmse)
        fold_results['train_mape'].append(train_mape)
        fold_results['test_mape'].append(test_mape)

    mean_test_rmse = np.mean(fold_results['test_rmse'])
    
    if mean_test_rmse < best_score:
        best_score = mean_test_rmse
        best_params_XGB_exloss = params
        best_cv_results = fold_results

print("XGB best parameters")
print(best_params_XGB_exloss)

train_r2 = np.mean(best_cv_results['train_r_squared'])
test_r2 = np.mean(best_cv_results['test_r_squared'])

train_rmse = np.mean(best_cv_results['train_rmse'])
test_rmse = np.mean(best_cv_results['test_rmse'])

train_mape = np.mean(best_cv_results['train_mape'])
test_mape = np.mean(best_cv_results['test_mape'])

print("XGB Inner CV Results")
print("===================================")
print("test r2")
print(test_r2)
print("train r2")
print(train_r2)
print("test rmse")
print(test_rmse)
print('train rmse')
print(train_rmse)
print("test mape")
print(test_mape)
print('train mape')
print(train_mape)

# # # # # # # # # # # # # # # # # # # # # # 
params_XGB_mse = {
    'max_depth': [10],
    'learning_rate': [0.1],
    #'n_estimators': [1500],
    #'num_leaves': [1023],
    'max_bin': [1000],
    'colsample_bytree': [0.8],
    'min_child_weight': [1],
    'reg_lambda': [50, 100, 1000],
    #'tree_method': ['gpu_hist']
    'objective': ['reg:squarederror'],
    'tree_method': ['hist']  # or 'exact', as applicable
}

#{'max_depth': 10, 'learning_rate': 0.1, 'max_bin': 1000, 'colsample_bytree': 1, 'min_child_weight': 1, 'reg_lambda': 10, 'objective': 'reg:squarederror', 'tree_method': 'hist'}

# Create all combinations of parameters
keys, values = zip(*params_XGB_mse.items())
param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

scoring = {'r_squared': 'r2', 'rmse': 'neg_root_mean_squared_error', 'mape': 'neg_mean_absolute_percentage_error'}
best_score = float('inf')
best_params_XGB_mse = None
best_cv_results = {}

# The number of splits for your cross-validation
num_splits = 5

for params in param_combinations:
    fold_results = {
        'train_r_squared': [],
        'test_r_squared': [],
        'train_rmse': [],
        'test_rmse': [],
        'train_mape': [],
        'test_mape': [],
    }
    
    for train_idx, test_idx in city_cv:
        dtrain = xgb.DMatrix(X_train_nested.iloc[train_idx], label=y_train_nested.iloc[train_idx].values.ravel())
        dtest = xgb.DMatrix(X_train_nested.iloc[test_idx], label=y_train_nested.iloc[test_idx].values.ravel())
        
        bst = xgb.train(params, dtrain, num_boost_round = 2000, obj=custom_loss)
        
        train_pred = bst.predict(dtrain)
        test_pred = bst.predict(dtest)
        
        # Calculate metrics for train and test datasets
        train_r2 = r2_score(y_train_nested.iloc[train_idx].values.ravel(), train_pred)
        test_r2 = r2_score(y_train_nested.iloc[test_idx].values.ravel(), test_pred)
        
        train_rmse = np.sqrt(mean_squared_error(y_train_nested.iloc[train_idx].values.ravel(), train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_train_nested.iloc[test_idx].values.ravel(), test_pred))
        
        train_mape = mean_absolute_percentage_error(y_train_nested.iloc[train_idx].values.ravel(), train_pred)
        test_mape = mean_absolute_percentage_error(y_train_nested.iloc[test_idx].values.ravel(), test_pred)
        
        fold_results['train_r_squared'].append(train_r2)
        fold_results['test_r_squared'].append(test_r2)
        fold_results['train_rmse'].append(train_rmse)
        fold_results['test_rmse'].append(test_rmse)
        fold_results['train_mape'].append(train_mape)
        fold_results['test_mape'].append(test_mape)

    mean_test_rmse = np.mean(fold_results['test_rmse'])
    
    if mean_test_rmse < best_score:
        best_score = mean_test_rmse
        best_params_XGB_mse = params
        best_cv_results = fold_results

print("XGB best parameters")
print(best_params_XGB_mse)

train_r2 = np.mean(best_cv_results['train_r_squared'])
test_r2 = np.mean(best_cv_results['test_r_squared'])

train_rmse = np.mean(best_cv_results['train_rmse'])
test_rmse = np.mean(best_cv_results['test_rmse'])

train_mape = np.mean(best_cv_results['train_mape'])
test_mape = np.mean(best_cv_results['test_mape'])

print("XGB Inner CV Results")
print("===================================")
print("test r2")
print(test_r2)
print("train r2")
print(train_r2)
print("test rmse")
print(test_rmse)
print('train rmse')
print(train_rmse)
print("test mape")
print(test_mape)
print('train mape')
print(train_mape)

# # # # # # # # # # # # # # # # # # # # # # 
trn_r2 = []
trn_rmse = []
cv_r2 = []
cv_rmse = []
dfs = []
train_dfs = []
eval_dfs = []

for i in range(10):
    train_data_XGB = train_concat[i].copy() #.drop(columns = ['date', 'grid_id', 'city', 'grid_id_12km', 'k_region', 'geometry'])
    y_trn = train_data_XGB.pop("pm25").to_frame() 
    X_trn = train_data_XGB
    train_df = pd.DataFrame({'date':X_trn['date'], 'grid_id':X_trn['grid_id'],
                            'k_region':X_trn['k_region'], 'y_trn': y_trn['pm25']})
    
    X_trn = X_trn.drop(columns = ['date', 'grid_id', 'grid_id_50km', 'k_region', 'geometry'])
    #print(f"X_trn.columns: {X_trn.columns}")
        
    test_data_XGB = test_concat[i].copy()#.drop(columns = ['date', 'grid_id', 'city', 'grid_id_12km', 'k_region', 'geometry'])
    y_val = test_data_XGB.pop("pm25").to_frame() 
    X_val = test_data_XGB
    
    eval_df = pd.DataFrame({'date':X_val['date'],'grid_id':X_val['grid_id'],'k_region': X_val['k_region'],
                            'y_val': y_val['pm25']})
    
    X_val = X_val.drop(columns = ['date', 'grid_id', 'grid_id_50km', 'k_region', 'geometry'])
    #print(X_val.columns)
   
    dtrain = xgb.DMatrix(X_trn, label=y_trn)
    dtest = xgb.DMatrix(X_val, label=y_val)
    print(f'dtrain shape: {(dtrain.num_row(), dtrain.num_col())}')
    print(f'dtest shape: {(dtest.num_row(), dtest.num_col())}')
    
    model_XGB = xgb.train(best_params_XGB_exloss, dtrain, num_boost_round = 2000, obj=custom_loss)          
    trn_y_pred = model_XGB.predict(dtrain)
    trn_score = r2_score(y_trn, trn_y_pred)
    print(f"Training R2: {trn_score}")
    trn_r2.append(trn_score)
    train_df['trn_y_pred'] = trn_y_pred
    train_dfs.append(train_df)
      
    trn_MSE = mean_squared_error(y_trn, trn_y_pred)
    trn_RMSE = math.sqrt(trn_MSE)
    trn_rmse.append(trn_RMSE)
    print(f"Training RMSE: {trn_RMSE}")
      
    y_pred = model_XGB.predict(dtest)
    eval_df['y_pred'] = y_pred
    eval_dfs.append(eval_df)
    
    R2 = r2_score(y_val, y_pred)
    print(f"CV R2: {R2}")
      
    MSE = mean_squared_error(y_val, y_pred)
    RMSE = math.sqrt(MSE)
    print(f"CV RMSE: {RMSE}")
    cv_r2.append(R2)
    cv_rmse.append(RMSE)

trn_r2_mean = np.mean(trn_r2)
trn_rmse_mean = np.mean(trn_rmse)
cv_r2_mean = np.mean(cv_r2)
cv_rmse_mean = np.mean(cv_rmse)

print("####### Exloss results #######")
                             
print(f"xgb train_r2_list: {trn_r2}")
print(f"xgb cv_r2_list: {cv_r2}")
print(f"xgb train_rmse_list: {trn_rmse}")
print(f"xgb cv_rmse_list: {cv_rmse}")
      
print(f"xgb cv_rmse: {cv_rmse_mean}")
print(f"xgb train_r2: {trn_r2_mean}")
print(f"xgb cv_r2: {cv_r2_mean}")
print(f"xgb train_rmse: {trn_rmse_mean}")
print(f"xgb cv_rmse: {cv_rmse_mean}")

train_dfs[0].to_csv(f"/scratch/users/akawano/pm_prediction/results/ML_full_model/{record}_fold_1_traindf.csv", index=False)
train_dfs[1].to_csv(f"/scratch/users/akawano/pm_prediction/results/ML_full_model/{record}_fold_2_traindf.csv", index=False)
train_dfs[2].to_csv(f"/scratch/users/akawano/pm_prediction/results/ML_full_model/{record}_fold_3_traindf.csv", index=False)
train_dfs[3].to_csv(f"/scratch/users/akawano/pm_prediction/results/ML_full_model/{record}_fold_4_traindf.csv", index=False)
train_dfs[4].to_csv(f"/scratch/users/akawano/pm_prediction/results/ML_full_model/{record}_fold_5_traindf.csv", index=False)
train_dfs[5].to_csv(f"/scratch/users/akawano/pm_prediction/results/ML_full_model/{record}_fold_6_traindf.csv", index=False)
train_dfs[6].to_csv(f"/scratch/users/akawano/pm_prediction/results/ML_full_model/{record}_fold_7_traindf.csv", index=False)
train_dfs[7].to_csv(f"/scratch/users/akawano/pm_prediction/results/ML_full_model/{record}_fold_8_traindf.csv", index=False)
train_dfs[8].to_csv(f"/scratch/users/akawano/pm_prediction/results/ML_full_model/{record}_fold_9_traindf.csv", index=False)
train_dfs[9].to_csv(f"/scratch/users/akawano/pm_prediction/results/ML_full_model/{record}_fold_10_traindf.csv", index=False)

eval_dfs[0].to_csv(f"/scratch/users/akawano/pm_prediction/results/ML_full_model/{record}_fold_1_evaldf.csv", index=False)
eval_dfs[1].to_csv(f"/scratch/users/akawano/pm_prediction/results/ML_full_model/{record}_fold_2_evaldf.csv", index=False)
eval_dfs[2].to_csv(f"/scratch/users/akawano/pm_prediction/results/ML_full_model/{record}_fold_3_evaldf.csv", index=False)
eval_dfs[3].to_csv(f"/scratch/users/akawano/pm_prediction/results/ML_full_model/{record}_fold_4_evaldf.csv", index=False)
eval_dfs[4].to_csv(f"/scratch/users/akawano/pm_prediction/results/ML_full_model/{record}_fold_5_evaldf.csv", index=False)
eval_dfs[5].to_csv(f"/scratch/users/akawano/pm_prediction/results/ML_full_model/{record}_fold_6_evaldf.csv", index=False)
eval_dfs[6].to_csv(f"/scratch/users/akawano/pm_prediction/results/ML_full_model/{record}_fold_7_evaldf.csv", index=False)
eval_dfs[7].to_csv(f"/scratch/users/akawano/pm_prediction/results/ML_full_model/{record}_fold_8_evaldf.csv", index=False)
eval_dfs[8].to_csv(f"/scratch/users/akawano/pm_prediction/results/ML_full_model/{record}_fold_9_evaldf.csv", index=False)
eval_dfs[9].to_csv(f"/scratch/users/akawano/pm_prediction/results/ML_full_model/{record}_fold_10_evaldf.csv", index=False)

# # # # # # # # # # # # # # # # # # # # # # 
trn_r2 = []
trn_rmse = []
cv_r2 = []
cv_rmse = []
dfs = []
train_dfs = []
eval_dfs = []

for i in range(10):
    train_data_XGB = train_concat[i].copy() #.drop(columns = ['date', 'grid_id', 'city', 'grid_id_12km', 'k_region', 'geometry'])
    y_trn = train_data_XGB.pop("pm25").to_frame() 
    X_trn = train_data_XGB
    train_df = pd.DataFrame({'date':X_trn['date'], 'grid_id':X_trn['grid_id'],
                            'k_region':X_trn['k_region'], 'y_trn': y_trn['pm25']})
    
    X_trn = X_trn.drop(columns = ['date', 'grid_id', 'grid_id_50km', 'k_region', 'geometry'])
    #print(f"X_trn.columns: {X_trn.columns}")
        
    test_data_XGB = test_concat[i].copy()#.drop(columns = ['date', 'grid_id', 'city', 'grid_id_12km', 'k_region', 'geometry'])
    y_val = test_data_XGB.pop("pm25").to_frame() 
    X_val = test_data_XGB
    
    eval_df = pd.DataFrame({'date':X_val['date'],'grid_id':X_val['grid_id'],'k_region': X_val['k_region'],
                            'y_val': y_val['pm25']})
    
    X_val = X_val.drop(columns = ['date', 'grid_id', 'grid_id_50km', 'k_region', 'geometry'])
    #print(X_val.columns)
   
    dtrain = xgb.DMatrix(X_trn, label=y_trn)
    dtest = xgb.DMatrix(X_val, label=y_val)
    print(f'dtrain shape: {(dtrain.num_row(), dtrain.num_col())}')
    print(f'dtest shape: {(dtest.num_row(), dtest.num_col())}')
    
    model_XGB = xgb.train(best_params_XGB_mse, dtrain, num_boost_round = 2000, obj=custom_loss)          
    trn_y_pred = model_XGB.predict(dtrain)
    trn_score = r2_score(y_trn, trn_y_pred)
    print(f"Training R2: {trn_score}")
    trn_r2.append(trn_score)
    train_df['trn_y_pred'] = trn_y_pred
    train_dfs.append(train_df)
      
    trn_MSE = mean_squared_error(y_trn, trn_y_pred)
    trn_RMSE = math.sqrt(trn_MSE)
    trn_rmse.append(trn_RMSE)
    print(f"Training RMSE: {trn_RMSE}")
      
    y_pred = model_XGB.predict(dtest)
    eval_df['y_pred'] = y_pred
    eval_dfs.append(eval_df)
    
    R2 = r2_score(y_val, y_pred)
    print(f"CV R2: {R2}")
      
    MSE = mean_squared_error(y_val, y_pred)
    RMSE = math.sqrt(MSE)
    print(f"CV RMSE: {RMSE}")
    cv_r2.append(R2)
    cv_rmse.append(RMSE)

trn_r2_mean = np.mean(trn_r2)
trn_rmse_mean = np.mean(trn_rmse)
cv_r2_mean = np.mean(cv_r2)
cv_rmse_mean = np.mean(cv_rmse)

print("####### MSE results #######")

print(f"xgb train_r2_list: {trn_r2}")
print(f"xgb cv_r2_list: {cv_r2}")
print(f"xgb train_rmse_list: {trn_rmse}")
print(f"xgb cv_rmse_list: {cv_rmse}")
      
print(f"xgb cv_rmse: {cv_rmse_mean}")
print(f"xgb train_r2: {trn_r2_mean}")
print(f"xgb cv_r2: {cv_r2_mean}")
print(f"xgb train_rmse: {trn_rmse_mean}")
print(f"xgb cv_rmse: {cv_rmse_mean}")

train_dfs[0].to_csv(f"/scratch/users/akawano/pm_prediction/results/ML_full_model/{record2}_fold_1_traindf.csv", index=False)
train_dfs[1].to_csv(f"/scratch/users/akawano/pm_prediction/results/ML_full_model/{record2}_fold_2_traindf.csv", index=False)
train_dfs[2].to_csv(f"/scratch/users/akawano/pm_prediction/results/ML_full_model/{record2}_fold_3_traindf.csv", index=False)
train_dfs[3].to_csv(f"/scratch/users/akawano/pm_prediction/results/ML_full_model/{record2}_fold_4_traindf.csv", index=False)
train_dfs[4].to_csv(f"/scratch/users/akawano/pm_prediction/results/ML_full_model/{record2}_fold_5_traindf.csv", index=False)
train_dfs[5].to_csv(f"/scratch/users/akawano/pm_prediction/results/ML_full_model/{record2}_fold_6_traindf.csv", index=False)
train_dfs[6].to_csv(f"/scratch/users/akawano/pm_prediction/results/ML_full_model/{record2}_fold_7_traindf.csv", index=False)
train_dfs[7].to_csv(f"/scratch/users/akawano/pm_prediction/results/ML_full_model/{record2}_fold_8_traindf.csv", index=False)
train_dfs[8].to_csv(f"/scratch/users/akawano/pm_prediction/results/ML_full_model/{record2}_fold_9_traindf.csv", index=False)
train_dfs[9].to_csv(f"/scratch/users/akawano/pm_prediction/results/ML_full_model/{record2}_fold_10_traindf.csv", index=False)

eval_dfs[0].to_csv(f"/scratch/users/akawano/pm_prediction/results/ML_full_model/{record2}_fold_1_evaldf.csv", index=False)
eval_dfs[1].to_csv(f"/scratch/users/akawano/pm_prediction/results/ML_full_model/{record2}_fold_2_evaldf.csv", index=False)
eval_dfs[2].to_csv(f"/scratch/users/akawano/pm_prediction/results/ML_full_model/{record2}_fold_3_evaldf.csv", index=False)
eval_dfs[3].to_csv(f"/scratch/users/akawano/pm_prediction/results/ML_full_model/{record2}_fold_4_evaldf.csv", index=False)
eval_dfs[4].to_csv(f"/scratch/users/akawano/pm_prediction/results/ML_full_model/{record2}_fold_5_evaldf.csv", index=False)
eval_dfs[5].to_csv(f"/scratch/users/akawano/pm_prediction/results/ML_full_model/{record2}_fold_6_evaldf.csv", index=False)
eval_dfs[6].to_csv(f"/scratch/users/akawano/pm_prediction/results/ML_full_model/{record2}_fold_7_evaldf.csv", index=False)
eval_dfs[7].to_csv(f"/scratch/users/akawano/pm_prediction/results/ML_full_model/{record2}_fold_8_evaldf.csv", index=False)
eval_dfs[8].to_csv(f"/scratch/users/akawano/pm_prediction/results/ML_full_model/{record2}_fold_9_evaldf.csv", index=False)
eval_dfs[9].to_csv(f"/scratch/users/akawano/pm_prediction/results/ML_full_model/{record2}_fold_10_evaldf.csv", index=False)


#############################################################################
'''
df = pd.read_csv("/scratch/users/akawano/pm_prediction/intermediate/ML_full_model/df_to_be_predicted.csv")

X_fin = df[['aot_daily', 'co_daily', 'v_wind', 'u_wind', 'rainfall', 'temp',
       'pressure', 'thermal_radiation', 'low_veg', 'high_veg', 'dewpoint_temp',
       'NO2_tropos', 'NO2_missing', 'aod', 'aod_missing', 'CO', 'CO_missing',
       'elevation', 'water', 'shurub', 'urban', 'forest', 'savannas', 'month',
       'day_of_year', 'monsoon', 'lon', 'lat',
       'wind_degree', 'RH', 'aot_rolling', 'co_rolling', 'omi_no2_rolling',
       'v_wind_rolling', 'u_wind_rolling', 'rainfall_rolling', 'temp_rolling',
       'wind_degree_rolling', 'RH_rolling', 'thermal_radiation_rolling',
       'dewpoint_temp_rolling', 'NO2_intped', 'aod_intped', 'CO_intped',
       'NO2_score', 'aod_score', 'CO_score', 'NO2_rolling', 'aod_rolling',
       'CO_rolling']].copy()


pred_XGB = model_XGB.predict(X_fin)
df['pm25_pred_xgb'] = pred_XGB
df.to_csv(f"/scratch/users/akawano/pm_prediction/intermediate/ML_full_model/pm25_pred_{record}.csv", index = False)

#df = pd.read_csv(f"/scratch/users/akawano/pm_prediction/pm25_pred_XGB_{record}.csv")
print(df.shape)
print(df['pm25_pred_xgb'].min())
#-9.183280584264288
print(df['pm25_pred_xgb'].max())
#852.0321429269533

df_negative = df.loc[df['pm25_pred_xgb'] < 0]
print(df_negative.shape)

#change negative predictions to 0
df.loc[df['pm25_pred_xgb'] < 0,'pm25_pred_xgb'] = 0
print(df.shape)
print(df['pm25_pred_xgb'].min())
print(df['pm25_pred_xgb'].max())

df.to_csv(f"/scratch/users/akawano/pm_prediction/intermediate/ML_full_model/pm25_pred_{record}_negatives_replaced.csv", index = False)

'''
