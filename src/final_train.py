from sklearn.model_selection import KFold
from cuml.preprocessing import TargetEncoder
import xgboost as xgb
import numpy as np

def final_train_and_predict(df_train, df_test, target, params, seed1=42):
    final_preds = np.zeros(df_test.shape[0])
    kf = KFold(n_splits=7, shuffle=True, random_state=seed1)

    for idx_train, idx_valid in kf.split(df_train):
        X_train, y_train = df_train.iloc[idx_train], target.iloc[idx_train]
        X_valid, y_valid = df_train.iloc[idx_valid], target.iloc[idx_valid]
        X_test = df_test[X_train.columns].copy()

        encoder = TargetEncoder(n_folds=5, seed=seed1, stat="mean")
        for col in df_train.columns[:20]:
            X_train[col+'_te'] = encoder.fit_transform(X_train[[col]], y_train)
            X_valid[col+'_te'] = encoder.transform(X_valid[[col]])
            X_test[col+'_te'] = encoder.transform(X_test[[col]])

        for col in df_train.columns[20:]:
            X_train[col] = encoder.fit_transform(X_train[[col]], y_train)
            X_valid[col] = encoder.transform(X_valid[[col]])
            X_test[col] = encoder.transform(X_test[[col]])

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dvalid = xgb.DMatrix(X_valid, label=y_valid)
        dtest = xgb.DMatrix(X_test)

        model = xgb.train(
            params,
            dtrain,
            num_boost_round=100000,
            evals=[(dtrain, 'train'), (dvalid, 'valid')],
            early_stopping_rounds=30,
            verbose_eval=500
        )

        final_preds += model.predict(dtest)

    final_preds /= 7
    return final_preds
