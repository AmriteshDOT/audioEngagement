from data_prep import load_prepare
from encoding_features import feature_eng
from optuna_tune import run_optuna
from final_train import final_train_and_predict


def main():
    df_train, df_test, target, df_org, df_train_orig, df_test_orig = load_prepare()
    df = feature_eng(df_train, df_train_orig)
    params = run_optuna(df_train, target)
    final_preds = final_train_and_predict(df_train, df_test, target, params)
    df_test["Listening_Time_minutes"] = final_preds


if __name__ == "__main__":
    main()
