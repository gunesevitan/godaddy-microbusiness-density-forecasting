import sys
import numpy as np
import pandas as pd

sys.path.append('..')
import settings


if __name__ == '__main__':

    df_train = pd.read_csv(settings.DATA / 'train.csv')
    settings.logger.info(f'train Shape: {df_train.shape} Memory Usage: {df_train.memory_usage().sum() / 1024 ** 2:.2f} MB')

    df_test = pd.read_csv(settings.DATA / 'test.csv')
    settings.logger.info(f'test Shape: {df_test.shape} Memory Usage: {df_test.memory_usage().sum() / 1024 ** 2:.2f} MB')

    df_revealed_test = pd.read_csv(settings.DATA / 'revealed_test.csv')
    settings.logger.info(f'revealed_test Shape: {df_revealed_test.shape} Memory Usage: {df_revealed_test.memory_usage().sum() / 1024 ** 2:.2f} MB')

    df_census_starter = pd.read_csv(settings.DATA / 'census_starter.csv')
    settings.logger.info(f'census_starter Shape: {df_census_starter.shape} Memory Usage: {df_census_starter.memory_usage().sum() / 1024 ** 2:.2f} MB')

    df_train = pd.concat((
        df_train,
        df_revealed_test
    ), axis=0, ignore_index=True)

    for df in [df_train, df_test]:

        df['cfips'] = df['cfips'].astype(np.uint32)
        if 'active' in df.columns:
            df['active'] = df['active'].astype(np.uint32)

        df['first_day_of_month'] = pd.to_datetime(df['first_day_of_month'])
        df['month'] = df['first_day_of_month'].dt.month.astype(np.uint8)
        df['year'] = df['first_day_of_month'].dt.year.astype(np.uint16)

    df_train.sort_values(by=['cfips', 'first_day_of_month'], inplace=True)
    df_train.reset_index(drop=True, inplace=True)

    for column in df_census_starter.columns:
        if df_census_starter[column].dtype == 'float64':
            df_census_starter[column] = df_census_starter[column].astype(np.float32)

        if df_census_starter[column].dtype == 'int64':
            df_census_starter[column] = df_census_starter[column].astype(np.uint32)

    # Restore missing county and state columns in test set
    df_test = df_test.merge(df_train.groupby('cfips').first().reset_index()[['cfips', 'county', 'state']], on='cfips', how='right')

    columns = [
        'row_id', 'cfips', 'county', 'state',
        'first_day_of_month', 'month', 'year',
        'microbusiness_density', 'active'
    ]

    df_train[columns].to_parquet(settings.DATA / 'train.parquet')
    settings.logger.info(f'train.parquet is saved to {settings.DATA}')

    df_test[columns[:-2]].to_parquet(settings.DATA / 'test.parquet')
    settings.logger.info(f'test.parquet is saved to {settings.DATA}')

    df_census_starter.to_parquet(settings.DATA / 'census_starter.parquet')
    settings.logger.info(f'census_starter.parquet is saved to {settings.DATA}')
