import pandas as pd
import numpy as np


def load_and_basic_prep():
    df_org = pd.read_csv("podcast_dataset.csv")
    df_train = pd.read_csv("train.csv")
    df_test = pd.read_csv("test.csv")

    df = pd.concat([df_train, df_test], ignore_index=True)
    df.drop(columns=["id"], inplace=True)

    # to remove errors if there
    df["Episode_Length_minutes"] = np.clip(df["Episode_Length_minutes"], 0, 120)
    df["Host_Popularity_percentage"] = np.clip(
        df["Host_Popularity_percentage"], 20, 100
    )
    df["Guest_Popularity_percentage"] = np.clip(
        df["Guest_Popularity_percentage"], 0, 100
    )
    df.loc[df["Number_of_Ads"] > 3, "Number_of_Ads"] = 0  # sus

    week = {
        "Monday": 1,
        "Tuesday": 2,
        "Wednesday": 3,
        "Thursday": 4,
        "Friday": 5,
        "Saturday": 6,
        "Sunday": 7,
    }
    df["Publication_Day"] = df["Publication_Day"].map(week)
    time = {"Morning": 1, "Afternoon": 2, "Evening": 3, "Night": 4}
    df["Publication_Time"] = df["Publication_Time"].map(time)

    sentiment = {"Negative": 1, "Neutral": 2, "Positive": 3}
    df["Episode_Sentiment"] = df["Episode_Sentiment"].map(sentiment)

    df["Episode_Title"] = (
        df["Episode_Title"].str.replace("Episode ", "", regex=True).astype(int)
    )

    return df, df_org, df_train, df_test
