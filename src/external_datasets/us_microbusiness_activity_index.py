import sys
import pathlib
import numpy as np
import pandas as pd

sys.path.append('..')
import settings


if __name__ == '__main__':

    external_raw_dataset_directory = pathlib.Path(settings.DATA / 'external' / 'raw' / 'VF_mai_bundle_Q222')
    external_processed_dataset_directory = pathlib.Path(settings.DATA / 'external' / 'processed')

    df_mai_cbsas = pd.read_csv(external_raw_dataset_directory / 'VF_mai_cbsas_Q222.csv')
    settings.logger.info(f'VF_mai_cbsas_Q222 Shape: {df_mai_cbsas.shape} - Memory Usage: {df_mai_cbsas.memory_usage().sum() / 1024 ** 2:.2f}')

    df_mai_national = pd.read_csv(external_raw_dataset_directory / 'VF_mai_national_Q222.csv')
    settings.logger.info(f'VF_mai_national_Q222 Shape: {df_mai_national.shape} - Memory Usage: {df_mai_national.memory_usage().sum() / 1024 ** 2:.2f}')

    df_mai_counties = pd.read_csv(external_raw_dataset_directory / 'VF_mai_counties_Q222.csv')
    settings.logger.info(f'VF_mai_counties_Q222 Shape: {df_mai_counties.shape} - Memory Usage: {df_mai_counties.memory_usage().sum() / 1024 ** 2:.2f}')

    df_mai_states = pd.read_csv(external_raw_dataset_directory / 'VF_mai_states_Q222.csv')
    settings.logger.info(f'VF_mai_states_Q222 Shape: {df_mai_states.shape} - Memory Usage: {df_mai_states.memory_usage().sum() / 1024 ** 2:.2f}')

    df_train = pd.read_parquet(settings.DATA / 'train.parquet')
    settings.logger.info(f'train Shape: {df_train.shape} - Memory Usage: {df_train.memory_usage().sum() / 1024 ** 2:.2f}')

    df_test = pd.read_parquet(settings.DATA / 'test.parquet')
    settings.logger.info(f'test Shape: {df_test.shape} - Memory Usage: {df_test.memory_usage().sum() / 1024 ** 2:.2f}')

    for df in [df_mai_cbsas, df_mai_counties, df_mai_states, df_mai_national]:
        df['date'] = pd.to_datetime(df['date'])
        for column in df.columns:
            if df[column].dtype == 'float64':
                df[column] = df[column].astype(np.float32)

    df_mai_counties = df_mai_counties.rename(columns={
        'date': 'first_day_of_month',
        'total_pop_20': 'county_total_pop_20',
        'MAI_composite': 'county_MAI_composite',
        'engagement': 'county_engagement',
        'participation': 'county_participation',
        'infrastructure': 'county_infrastructure',
    }).drop(columns=['county_name'])
    df_mai_counties.to_parquet(external_processed_dataset_directory / 'county_microbusiness_activity_index.parquet')
    settings.logger.info(f'county_microbusiness_activity_index.parquet is saved to {external_processed_dataset_directory}')

    df_mai_states = df_mai_states.rename(columns={
        'date': 'first_day_of_month',
        'total_pop_20': 'state_total_pop_20',
        'MAI_composite': 'state_MAI_composite',
        'engagement': 'state_engagement',
        'participation': 'state_participation',
        'infrastructure': 'state_infrastructure',
    }).drop(columns=['state_abbrev', 'state_name'])
    df_mai_states.to_parquet(external_processed_dataset_directory / 'state_microbusiness_activity_index.parquet')
    settings.logger.info(f'state_microbusiness_activity_index.parquet is saved to {external_processed_dataset_directory}')

    df_mai_national = df_mai_national.rename(columns={
        'date': 'first_day_of_month',
        'MAI_composite': 'national_MAI_composite',
        'engagement': 'national_engagement',
        'participation': 'national_participation',
        'infrastructure': 'national_infrastructure',
    })
    df_mai_national.to_parquet(external_processed_dataset_directory / 'national_microbusiness_activity_index.parquet')
    settings.logger.info(f'national_microbusiness_activity_index.parquet is saved to {external_processed_dataset_directory}')
