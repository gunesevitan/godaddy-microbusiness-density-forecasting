import sys
import pathlib
import numpy as np
import pandas as pd

sys.path.append('..')
import settings


if __name__ == '__main__':

    external_raw_dataset_directory = pathlib.Path(settings.DATA / 'external' / 'raw' / 'census')
    external_processed_dataset_directory = pathlib.Path(settings.DATA / 'external' / 'processed')

    census_columns = [
        'GEO_ID', 'S0101_C01_026E'
    ]

    df_acs_5y_2017 = pd.read_csv(external_raw_dataset_directory / 'ACSST5Y2017.S0101-Data.csv', usecols=census_columns).iloc[1:]
    settings.logger.info(f'ACSST5Y2017.S0101-Data Shape: {df_acs_5y_2017.shape} - Memory Usage: {df_acs_5y_2017.memory_usage().sum() / 1024 ** 2:.2f}')

    df_acs_5y_2018 = pd.read_csv(external_raw_dataset_directory / 'ACSST5Y2018.S0101-Data.csv', usecols=census_columns).iloc[1:]
    settings.logger.info(f'ACSST5Y2018.S0101-Data Shape: {df_acs_5y_2018.shape} - Memory Usage: {df_acs_5y_2018.memory_usage().sum() / 1024 ** 2:.2f}')

    df_acs_5y_2019 = pd.read_csv(external_raw_dataset_directory / 'ACSST5Y2019.S0101-Data.csv', usecols=census_columns).iloc[1:]
    settings.logger.info(f'ACSST5Y2019.S0101-Data Shape: {df_acs_5y_2019.shape} - Memory Usage: {df_acs_5y_2019.memory_usage().sum() / 1024 ** 2:.2f}')

    df_acs_5y_2020 = pd.read_csv(external_raw_dataset_directory / 'ACSST5Y2020.S0101-Data.csv', usecols=census_columns).iloc[1:]
    settings.logger.info(f'ACSST5Y2020.S0101-Data Shape: {df_acs_5y_2020.shape} - Memory Usage: {df_acs_5y_2020.memory_usage().sum() / 1024 ** 2:.2f}')

    df_acs_5y_2021 = pd.read_csv(external_raw_dataset_directory / 'ACSST5Y2021.S0101-Data.csv', usecols=census_columns).iloc[1:]
    settings.logger.info(f'ACSST5Y2021.S0101-Data Shape: {df_acs_5y_2021.shape} - Memory Usage: {df_acs_5y_2021.memory_usage().sum() / 1024 ** 2:.2f}')

    for df in [df_acs_5y_2017, df_acs_5y_2018, df_acs_5y_2019, df_acs_5y_2020, df_acs_5y_2021]:
        df['S0101_C01_026E'] = df['S0101_C01_026E'].astype(int)
        df['cfips'] = df['GEO_ID'].apply(lambda x: int(x.split('US')[-1]))
        df['cfips'] = df['cfips'].astype(int)

    df_train = pd.read_parquet(settings.DATA / 'train.parquet')
    settings.logger.info(f'train Shape: {df_train.shape} - Memory Usage: {df_train.memory_usage().sum() / 1024 ** 2:.2f}')

    df_test = pd.read_parquet(settings.DATA / 'test.parquet')
    settings.logger.info(f'test Shape: {df_test.shape} - Memory Usage: {df_test.memory_usage().sum() / 1024 ** 2:.2f}')

    df_census_starter = pd.read_parquet(settings.DATA / 'census_starter.parquet')
    settings.logger.info(f'census_starter Shape: {df_census_starter.shape} - Memory Usage: {df_census_starter.memory_usage().sum() / 1024 ** 2:.2f}')