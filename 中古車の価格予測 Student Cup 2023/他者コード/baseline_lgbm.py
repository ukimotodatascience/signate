import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import category_encoders as ce

import lightgbm as lgb

from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import KFold

import warnings
warnings.filterwarnings('ignore')

# load data
train_df = pd.read_csv("./data/train.csv")
test_df = pd.read_csv("./data/test.csv")

# preprocessing
cat_cols = ["region", "manufacturer", "condition", "fuel", "title_status", "cylinders",
            "transmission", "drive", "size", "type", "paint_color", "state"]

## cat -> count encoding
def count_encoder(df, cat_cols):
    ce_ord = ce.CountEncoder(cols = cat_cols)
    encoded_df = ce_ord.fit_transform(df)
    df = df.drop(columns=cat_cols)
    
    return encoded_df

## target log transform
def log_trainsform(df, cols):
    return df[cols]

train_df = count_encoder(train_df, cat_cols)
test_df = count_encoder(test_df, cat_cols)

# model
features = [c for c in train_df.columns if c not in ["id", "price"]]
target = train_df["price"]

param = {
    'bagging_freq': 5,
    'bagging_fraction': 0.4,
    'boost_from_average':'false',
    'boost': 'gbdt',
    'feature_fraction': 0.05,
    'learning_rate': 0.01,
    'max_depth': -1,  
    'metric':'mape',
    'min_data_in_leaf': 80,
    'min_sum_hessian_in_leaf': 10.0,
    'num_leaves': 13,
    'num_threads': 8,
    'tree_learner': 'serial',
    'objective': 'regression', 
    'verbosity': -1
}

folds = KFold(n_splits=10, shuffle=True, random_state=44000)
oof = np.zeros(len(train_df))
predictions = np.zeros(len(test_df))
feature_importance_df = pd.DataFrame()

for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_df.values, target.values)):
    print("Fold {}".format(fold_+1))
    trn_data = lgb.Dataset(train_df.iloc[trn_idx][features], label=target.iloc[trn_idx])
    val_data = lgb.Dataset(train_df.iloc[val_idx][features], label=target.iloc[val_idx])

    num_round = 1000000
    rgl = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], 
                    callbacks=[lgb.early_stopping(stopping_rounds=3000, verbose=True),
                               lgb.log_evaluation(1000)]
                   )
    oof[val_idx] = rgl.predict(train_df.iloc[val_idx][features], num_iteration=rgl.best_iteration)
    
    fold_importance_df = pd.DataFrame()
    fold_importance_df["Feature"] = features
    fold_importance_df["importance"] = rgl.feature_importance()
    fold_importance_df["fold"] = fold_ + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    
    predictions += rgl.predict(test_df[features], num_iteration=rgl.best_iteration) / folds.n_splits

print("CV score: {:<8.5f}".format(mean_absolute_percentage_error(target, oof)))

# submission file
sub_df = pd.DataFrame({"id":test_df["id"].values})
sub_df["price"] = np.exp(predictions)
sub_df.to_csv("./submission/baseline_lgbm.csv", index=False, header=False)