from sklearn.preprocessing import LabelEncoder
import numpy as np
from itertools import combinations


def apply_encoding_and_features(df, original_df_train):
    le = LabelEncoder()
    for col in df.select_dtypes("object").columns:
        df[col] = le.fit_transform(df[col]) + 1

    for col in ["Episode_Length_minutes"]:
        df[col + "_sqrt"] = np.sqrt(df[col])
        df[col + "_squared"] = df[col] ** 2

    cols = [
        "Episode_Sentiment",
        "Genre",
        "Publication_Day",
        "Podcast_Name",
        "Episode_Title",
        "Guest_Popularity_percentage",
        "Host_Popularity_percentage",
        "Number_of_Ads",
    ]
    for col in cols:
        means = original_df_train.groupby(col)["Listening_Time_minutes"].mean()
        df[col + "_EP"] = df[col].map(means)

    def process_combinations_fast(
        df, columns_to_encode, pair_size, max_batch_size=2000
    ):
        str_df = df[columns_to_encode].astype(str)
        le = LabelEncoder()
        for r in pair_size:
            combos_iter = combinations(columns_to_encode, r)
            for cols in combos_iter:
                new_name = "+".join(cols)
                result = str_df[cols[0]].copy()
                for col in cols[1:]:
                    result += str_df[col]
                df[new_name] = le.fit_transform(result) + 1
        return df

    df = process_combinations_fast(
        df,
        [
            "Episode_Length_minutes",
            "Episode_Title",
            "Publication_Time",
            "Host_Popularity_percentage",
            "Number_of_Ads",
            "Episode_Sentiment",
            "Publication_Day",
            "Podcast_Name",
            "Genre",
            "Guest_Popularity_percentage",
        ],
        [2, 3, 5, 7],
    )

    df = df.astype("float32")
    return df
