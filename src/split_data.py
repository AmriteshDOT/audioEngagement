def split_train_test(df, df_test_original):
    df_train = df.iloc[: -len(df_test_original)]
    df_test = df.iloc[-len(df_test_original) :].reset_index(drop=True)
    df_train = df_train[df_train["Listening_Time_minutes"].notnull()]
    target = df_train.pop("Listening_Time_minutes")
    df_test.drop(columns=["Listening_Time_minutes"], inplace=True)
    return df_train, df_test, target
