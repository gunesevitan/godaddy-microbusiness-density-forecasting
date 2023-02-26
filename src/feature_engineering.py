import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

import settings


def extract_datetime_features(df, datetime_column):

    """
    Extract features from given datetime column

    Parameters
    ----------
    df: pandas.DataFrame
        Dataframe with given datetime column

    datetime_column: str
        Name of the datetime column

    Returns
    -------
    df: pandas.DataFrame
        Dataframe with datetime features
    """

    df[datetime_column] = pd.to_datetime(df[datetime_column])
    df['month'] = df['first_day_of_month'].dt.month.astype(np.uint8)
    df['year'] = df['first_day_of_month'].dt.year.astype(np.uint16)

    return df


def merge_census_features(df):

    """
    Merge processed census features

    Parameters
    ----------
    df: pandas.DataFrame
        Dataframe with given cfips and first_day_of_year columns

    Returns
    -------
    df: pandas.DataFrame
        Dataframe with census features
    """

    df_census = pd.read_parquet(settings.DATA / 'census.parquet')
    df_census['first_day_of_year'] = pd.to_datetime(df_census['first_day_of_year'])
    df_census['year'] = df_census['first_day_of_year'].dt.year.astype(np.uint16)
    df = df.merge(df_census.drop(columns=['first_day_of_year']), on=['cfips', 'year'], how='left')

    df_acs_5y = pd.read_parquet(settings.DATA / 'external' / 'processed' / 'acs_5y.parquet')
    df_acs_5y['year'] = df_acs_5y['first_day_of_year'].dt.year.astype(np.uint16)
    df = df.merge(df_acs_5y.drop(columns=['first_day_of_year']), on=['cfips', 'year'], how='left')

    return df


def merge_godaddy_features(df):

    """
    Merge processed GoDaddy features

    Parameters
    ----------
    df: pandas.DataFrame
        Dataframe with given cfips and first_day_of_year columns

    Returns
    -------
    df: pandas.DataFrame
        Dataframe with GoDaddy features
    """

    df_county_industry_commerce = pd.read_parquet(settings.DATA / 'external' / 'processed' / 'county_industry_commerce.parquet')
    df = df.merge(df_county_industry_commerce, on=['cfips', 'first_day_of_month'], how='left')

    df_county_microbusiness_activity_index = pd.read_parquet(settings.DATA / 'external' / 'processed' / 'county_microbusiness_activity_index.parquet')
    df = df.merge(df_county_microbusiness_activity_index.drop(columns=['county_total_pop_20']), on=['cfips', 'first_day_of_month'], how='left')

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

    categorical_columns = ['state']

    for categorical_column in categorical_columns:
        df[categorical_column] = LabelEncoder().fit_transform(df[categorical_column].values.reshape(-1, 1))

    return df


def create_lag_features(df):

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

    for feature in [
        'microbusiness_density', 'active',
        'county_orders_rank', 'county_merchants_rank', 'county_gmv_rank', 'county_total_pop_20',
        'county_MAI_composite',  'county_engagement', 'county_participation', 'county_infrastructure'
    ]:
        for shift in [5, 6, 7, 8, 9, 10]:
            df[f'{feature}_shift_{shift}'] = df.groupby('cfips')[feature].shift(periods=shift).astype(np.float32)

        df[f'{feature}_shift_5_diff_1'] = df.groupby('cfips')[f'{feature}_shift_5'].diff(periods=1).astype(np.float32)

    for feature in ['county_MAI_composite',  'county_engagement', 'county_participation', 'county_infrastructure']:
        for shift in [13]:
            df[f'{feature}_shift_{shift}'] = df.groupby('cfips')[feature].shift(periods=shift).astype(np.float32)

    return df


def create_aggregation_features(df):

    """
    Create aggregation features

    Parameters
    ----------
    df: pandas.DataFrame
        Dataframe with cfips column

    Returns
    -------
    df: pandas.DataFrame
        Dataframe with aggregation features
    """

    df['cfips_cumcount'] = df.groupby('cfips').transform('cumcount')

    return df


def create_features(df, datetime_column='first_day_of_month'):

    """
    Create features sequentially

    Parameters
    ----------
    df: pandas.DataFrame
        Raw dataframe

    datetime_column: str
        Name of the datetime column

    Returns
    -------
    df: pandas.DataFrame
        Dataframe with features
    """

    df = extract_datetime_features(df=df, datetime_column=datetime_column)
    df = merge_census_features(df=df)
    df = merge_godaddy_features(df=df)

    df = encode_categorical_features(df=df)
    df = create_lag_features(df=df)
    df = create_aggregation_features(df=df)

    return df
