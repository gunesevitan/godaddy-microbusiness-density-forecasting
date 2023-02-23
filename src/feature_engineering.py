import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

import settings


def merge_census_features(df):

    df = df.merge(pd.read_parquet(settings.DATA / 'census_starter.parquet'), on='cfips', how='left')

    return df

def encode_categorical_features(df):

    """
    Label encode categorical features

    Parameters
    ----------
    df: pandas.DataFrame
        Dataframe with given categorical columns

    Returns
    -------
    df: pandas.DataFrame
        Dataframe with encoded categorical features
    """

    categorical_columns = [
        'state'
    ]
    df[categorical_columns] = LabelEncoder().fit_transform(categorical_columns)

    return df


def create_lag_features(df):

    df['microbusiness_density_shift_5'] = df.groupby('cfips')['microbusiness_density'].shift(periods=5).astype(np.float32)
    df['microbusiness_density_shift_6'] = df.groupby('cfips')['microbusiness_density'].shift(periods=6).astype(np.float32)
    df['microbusiness_density_shift_7'] = df.groupby('cfips')['microbusiness_density'].shift(periods=7).astype(np.float32)
    df['microbusiness_density_shift_8'] = df.groupby('cfips')['microbusiness_density'].shift(periods=8).astype(np.float32)
    df['microbusiness_density_shift_9'] = df.groupby('cfips')['microbusiness_density'].shift(periods=9).astype(np.float32)
    df['microbusiness_density_shift_10'] = df.groupby('cfips')['microbusiness_density'].shift(periods=10).astype(np.float32)
    df['microbusiness_density_shift_11'] = df.groupby('cfips')['microbusiness_density'].shift(periods=11).astype(np.float32)
    df['microbusiness_density_shift_12'] = df.groupby('cfips')['microbusiness_density'].shift(periods=12).astype(np.float32)
    df['active_shift_5'] = df.groupby('cfips')['active'].shift(periods=5).astype(np.float32)
    df['active_shift_6'] = df.groupby('cfips')['active'].shift(periods=6).astype(np.float32)

    df['microbusiness_density_shift_5_diff_1'] = df.groupby('cfips')['microbusiness_density_shift_5'].diff(periods=1).astype(np.float32)
    df['microbusiness_density_shift_5_pct_change_1'] = df.groupby('cfips')['microbusiness_density_shift_5'].pct_change(periods=1).astype(np.float32)

    df['microbusiness_density_shift_5_rolling_mean_3'] = df.groupby('cfips')['microbusiness_density_shift_5'].ewm(com=0.001, min_periods=1).mean().values
    #df['microbusiness_density_shift_5_rolling_std_3'] = df.groupby('cfips')['microbusiness_density_shift_5'].ewm(com=5, min_periods=1).std().values
    #df['microbusiness_density_shift_5_rolling_min_3'] = df.groupby('cfips')['microbusiness_density_shift_5'].ewm(com=5, min_periods=1).min().values
    #df['microbusiness_density_shift_5_rolling_max_3'] = df.groupby('cfips')['microbusiness_density_shift_5'].ewm(com=5, min_periods=1).max().values

    return df


def create_aggregation_features(df):

    df['cfips_cumcount'] = df.groupby('cfips').transform('cumcount')

    return df


def _scale_features(df, features):

    df[features] = StandardScaler().fit_transform(df[features])

    return df


def create_features(df):

    df = merge_census_features(df=df)

    df = encode_categorical_features(df=df)
    df = create_lag_features(df=df)
    df = create_aggregation_features(df=df)

    return df
