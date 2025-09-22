from sklearn.model_selection import KFold
import optuna
from cuml.preprocessing import TargetEncoder
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import numpy as np

def run_optuna(df_train, target, seed1=42):
    cv = KFold(n_splits=5, shuffle=True, random_state=seed1)

    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "device": "cuda",
        "seed": seed1,
    }

    def objective(trial):
        tuned_params = {
            "max_depth": trial.suggest_int("max_depth", 6, 20),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 100),
            "reg_alpha": trial.suggest_float("reg_alpha", 0, 10),
            "reg_lambda": trial.suggest_float("reg_lambda", 0, 10),
            "gamma": trial.suggest_float("gamma", 0, 5),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "colsample_bynode": trial.suggest_float("colsample_bynode", 0.3, 1.0),
        }
        full_params = params.copy()
        full_params.update(tuned_params)
        fold_scores = []
        for train_idx, valid_idx in cv.split(df_train):
            X_train, y_train = df_train.iloc[train_idx], target.iloc[train_idx]
            X_valid, y_valid = df_train.iloc[valid_idx], target.iloc[valid_idx]

            encoder = TargetEncoder(n_folds=5, seed=seed1, stat="mean")
            for col in df_train.columns[:20]:
                X_train[col + "_te"] = encoder.fit_transform(X_train[[col]], y_train)
                X_valid[col + "_te"] = encoder.transform(X_valid[[col]])

            dtrain = xgb.DMatrix(X_train, label=y_train)
            dvalid = xgb.DMatrix(X_valid, label=y_valid)

            model = xgb.train(
                full_params,
                dtrain,
                num_boost_round=10000,
                evals=[(dvalid, "valid")],
                early_stopping_rounds=50,
                verbose_eval=False,
            )
            preds = model.predict(dvalid)
            fold_scores.append(mean_squared_error(y_valid, preds, squared=False))
        return np.mean(fold_scores)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=30)

    params.update(study.best_params)
    return params
