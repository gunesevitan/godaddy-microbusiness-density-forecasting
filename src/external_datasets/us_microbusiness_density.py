import sys
import pathlib
import numpy as np
import pandas as pd

sys.path.append('..')
import settings


if __name__ == '__main__':

    external_raw_dataset_directory = pathlib.Path(settings.DATA / 'external' / 'raw' / 'VF_md_bundle_Q222')
    external_processed_dataset_directory = pathlib.Path(settings.DATA / 'external' / 'processed')

    df_md_cbsas = pd.read_csv(external_raw_dataset_directory / 'VF_md_cbsas_Q222.csv')
    settings.logger.info(f'VF_md_cbsas_Q222 Shape: {df_md_cbsas.shape} - Memory Usage: {df_md_cbsas.memory_usage().sum() / 1024 ** 2:.2f}')

    df_md_cities = pd.read_csv(external_raw_dataset_directory / 'VF_md_cities_Q222.csv')
    settings.logger.info(f'VF_md_cities_Q222 Shape: {df_md_cities.shape} - Memory Usage: {df_md_cities.memory_usage().sum() / 1024 ** 2:.2f}')

    df_md_counties = pd.read_csv(external_raw_dataset_directory / 'VF_md_counties_Q222.csv')
    settings.logger.info(f'VF_md_counties_Q222 Shape: {df_md_counties.shape} - Memory Usage: {df_md_counties.memory_usage().sum() / 1024 ** 2:.2f}')

    df_md_states = pd.read_csv(external_raw_dataset_directory / 'VF_md_states_Q222.csv')
    settings.logger.info(f'VF_md_states_Q222 Shape: {df_md_states.shape} - Memory Usage: {df_md_states.memory_usage().sum() / 1024 ** 2:.2f}')

    df_train = pd.read_parquet(settings.DATA / 'train.parquet')
    settings.logger.info(f'train Shape: {df_train.shape} - Memory Usage: {df_train.memory_usage().sum() / 1024 ** 2:.2f}')

    df_test = pd.read_parquet(settings.DATA / 'test.parquet')
    settings.logger.info(f'test Shape: {df_test.shape} - Memory Usage: {df_test.memory_usage().sum() / 1024 ** 2:.2f}')

    # Select columns of each year separately
    active_2019_columns = [column for column in df_md_cities.columns if column.startswith('active') and column.endswith('19')]
    active_2020_columns = [column for column in df_md_cities.columns if column.startswith('active') and column.endswith('20')]
    active_2021_columns = [column for column in df_md_cities.columns if column.startswith('active') and column.endswith('21')]
    active_2022_columns = [column for column in df_md_cities.columns if column.startswith('active') and column.endswith('22')]
    md_2019_columns = [column for column in df_md_cities.columns if column.startswith('md') and column.endswith('19')]
    md_2020_columns = [column for column in df_md_cities.columns if column.startswith('md') and column.endswith('20')]
    md_2021_columns = [column for column in df_md_cities.columns if column.startswith('md') and column.endswith('21')]
    md_2022_columns = [column for column in df_md_cities.columns if column.startswith('md') and column.endswith('22')]
    column_groups = [
        active_2019_columns, active_2020_columns, active_2021_columns, active_2022_columns,
        md_2019_columns, md_2020_columns, md_2021_columns, md_2022_columns,
    ]
    columns_group_names = [
        'active_2019', 'active_2020', 'active_2021', 'active_2022',
        'md_2019', 'md_2020', 'md_2021', 'md_2022',
    ]
    # Aggregate microbusiness density and active on years for every dataset
    for columns_group_name, column_group in zip(columns_group_names, column_groups):
        for df in [df_md_cbsas, df_md_cities, df_md_counties, df_md_states]:
            df[f'{columns_group_name}_mean'] = df[column_group].mean(axis=1)
            df[f'{columns_group_name}_std'] = df[column_group].std(axis=1)
            df[f'{columns_group_name}_min'] = df[column_group].min(axis=1)
            df[f'{columns_group_name}_max'] = df[column_group].max(axis=1)
            df[f'{columns_group_name}_sum'] = df[column_group].sum(axis=1)

    # Aggregate features on state groups and merge it to state data
    df_md_cities.drop(columns=['city_id', 'city'], inplace=True)
    df_md_cities_aggregations = df_md_cities.groupby('state_abbrev').agg(['mean', 'std', 'min', 'max', 'sum'])
    df_md_cities_aggregations.columns = 'state_' + df_md_cities_aggregations.columns.map('_'.join).str.strip('_')
    df_md_cities_aggregations.reset_index(inplace=True)

    for column in df_md_cities_aggregations.columns:
        if df_md_cities_aggregations[column].dtype == 'float64':
            df_md_cities_aggregations[column] = df_md_cities_aggregations[column].astype(np.float32)

    for column in df_md_states.columns:
        if df_md_states[column].dtype == 'float64':
            df_md_states[column] = df_md_states[column].astype(np.float32)

    df_md_states.columns = [f'state_{column}' for column in df_md_states.columns]
    df_md_states.rename(columns={'state_state_abbrev': 'state_abbrev', 'state_fips': 'fips'}, inplace=True)
    df_md_states = df_md_states.merge(df_md_cities_aggregations, on='state_abbrev', how='left')
    df_md_states = df_md_states.drop(columns=['state_abbrev']).sort_values(by='fips', ascending=True).reset_index(drop=True)
    df_md_states.to_parquet(external_processed_dataset_directory / 'state_microbusiness_densities.parquet')
    settings.logger.info(f'state_microbusiness_densities.parquet is saved to {external_processed_dataset_directory}')

    for column in df_md_counties.columns:
        if df_md_counties[column].dtype == 'float64':
            df_md_counties[column] = df_md_counties[column].astype(np.float32)

    # Drop counties that doesn't exist in training and test sets
    df_md_counties = df_md_counties.loc[df_md_counties['cfips'].isin(df_train['cfips'].unique())]
    df_md_counties = df_md_counties.sort_values(by='cfips', ascending=True).reset_index(drop=True)
    df_md_counties['cfips'] = df_md_counties['cfips'].astype('int')
    df_md_counties['total_pop_20'] = df_md_counties['total_pop_20'].astype('int')
    df_md_counties.drop(columns=['county', 'state'], inplace=True)
    df_md_counties.columns = [f'county_{column}' for column in df_md_counties.columns]
    df_md_counties.rename(columns={'county_cfips': 'cfips'}, inplace=True)
    df_md_counties.to_parquet(external_processed_dataset_directory / 'county_microbusiness_densities.parquet')
    settings.logger.info(f'county_microbusiness_densities.parquet is saved to {external_processed_dataset_directory}')
