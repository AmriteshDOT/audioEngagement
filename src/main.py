from data_prep import load_and_basic_prep
from encoding_features import apply_encoding_and_features
from split_data import split_train_test
from optuna_tune import run_optuna
from final_train import final_train_and_predict

def main():
    df, df_org, df_train_orig, df_test_orig = load_and_basic_prep()
    df = apply_encoding_and_features(df, df_train_orig)
    df_train, df_test, target = split_train_test(df, df_test_orig)
    params = run_optuna(df_train, target)
    final_preds = final_train_and_predict(df_train, df_test, target, params)
    df_test["Listening_Time_minutes"] = final_preds

if __name__ == "__main__":
    main()
