import sys
import pathlib
import numpy as np
import pandas as pd

sys.path.append('..')
import settings


if __name__ == '__main__':

    external_raw_dataset_directory = pathlib.Path(settings.DATA / 'external' / 'raw' / 'VF_indcom_bundle_Q222')
    external_processed_dataset_directory = pathlib.Path(settings.DATA / 'external' / 'processed')

    df_indcom_cbsas = pd.read_csv(external_raw_dataset_directory / 'VF_indcom_cbsa_Q222.csv')
    settings.logger.info(f'VF_indcom_cbsas_Q222 Shape: {df_indcom_cbsas.shape} - Memory Usage: {df_indcom_cbsas.memory_usage().sum() / 1024 ** 2:.2f}')

    df_indcom_cities = pd.read_csv(external_raw_dataset_directory / 'VF_indcom_cities_Q222.csv')
    settings.logger.info(f'VF_indcom_cities_Q222 Shape: {df_indcom_cities.shape} - Memory Usage: {df_indcom_cities.memory_usage().sum() / 1024 ** 2:.2f}')

    df_indcom_counties = pd.read_csv(external_raw_dataset_directory / 'VF_indcom_counties_Q222.csv')
    settings.logger.info(f'VF_indcom_counties_Q222 Shape: {df_indcom_counties.shape} - Memory Usage: {df_indcom_counties.memory_usage().sum() / 1024 ** 2:.2f}')

    df_indcom_states = pd.read_csv(external_raw_dataset_directory / 'VF_indcom_states_Q222.csv')
    settings.logger.info(f'VF_indcom_states_Q222 Shape: {df_indcom_states.shape} - Memory Usage: {df_indcom_states.memory_usage().sum() / 1024 ** 2:.2f}')

    df_train = pd.read_parquet(settings.DATA / 'train.parquet')
    settings.logger.info(f'train Shape: {df_train.shape} - Memory Usage: {df_train.memory_usage().sum() / 1024 ** 2:.2f}')

    df_test = pd.read_parquet(settings.DATA / 'test.parquet')
    settings.logger.info(f'test Shape: {df_test.shape} - Memory Usage: {df_test.memory_usage().sum() / 1024 ** 2:.2f}')

    # Select columns of each year separately
    orders_rank_2019_columns = [column for column in df_indcom_cities.columns if column.startswith('orders_rank') and column.endswith('19')]
    orders_rank_2020_columns = [column for column in df_indcom_cities.columns if column.startswith('orders_rank') and column.endswith('20')]
    orders_rank_2021_columns = [column for column in df_indcom_cities.columns if column.startswith('orders_rank') and column.endswith('21')]
    orders_rank_2022_columns = [column for column in df_indcom_cities.columns if column.startswith('orders_rank') and column.endswith('22')]
    merchants_rank_2019_columns = [column for column in df_indcom_cities.columns if column.startswith('merchants_rank') and column.endswith('19')]
    merchants_rank_2020_columns = [column for column in df_indcom_cities.columns if column.startswith('merchants_rank') and column.endswith('20')]
    merchants_rank_2021_columns = [column for column in df_indcom_cities.columns if column.startswith('merchants_rank') and column.endswith('21')]
    merchants_rank_2022_columns = [column for column in df_indcom_cities.columns if column.startswith('merchants_rank') and column.endswith('22')]
    gmv_rank_2019_columns = [column for column in df_indcom_cities.columns if column.startswith('gmv_rank') and column.endswith('19')]
    gmv_rank_2020_columns = [column for column in df_indcom_cities.columns if column.startswith('gmv_rank') and column.endswith('20')]
    gmv_rank_2021_columns = [column for column in df_indcom_cities.columns if column.startswith('gmv_rank') and column.endswith('21')]
    gmv_rank_2022_columns = [column for column in df_indcom_cities.columns if column.startswith('gmv_rank') and column.endswith('22')]
    top3industries_2019_columns = [column for column in df_indcom_cities.columns if column.startswith('top3industries') and column.endswith('19')]
    top3industries_2020_columns = [column for column in df_indcom_cities.columns if column.startswith('top3industries') and column.endswith('20')]
    top3industries_2021_columns = [column for column in df_indcom_cities.columns if column.startswith('top3industries') and column.endswith('21')]
    top3industries_2022_columns = [column for column in df_indcom_cities.columns if column.startswith('top3industries') and column.endswith('22')]
    avg_traffic_2019_columns = [column for column in df_indcom_cities.columns if column.startswith('avg_traffic') and column.endswith('19')]
    avg_traffic_2020_columns = [column for column in df_indcom_cities.columns if column.startswith('avg_traffic') and column.endswith('20')]
    avg_traffic_2021_columns = [column for column in df_indcom_cities.columns if column.startswith('avg_traffic') and column.endswith('21')]
    avg_traffic_2022_columns = [column for column in df_indcom_cities.columns if column.startswith('avg_traffic') and column.endswith('22')]

    # Drop categorical columns
    for df in [df_indcom_cbsas, df_indcom_cities, df_indcom_counties, df_indcom_states]:
        df.drop(columns=top3industries_2019_columns + top3industries_2020_columns + top3industries_2021_columns + top3industries_2022_columns, inplace=True)

    column_groups = [
        orders_rank_2019_columns, orders_rank_2020_columns, orders_rank_2021_columns, orders_rank_2022_columns,
        merchants_rank_2019_columns, merchants_rank_2020_columns, merchants_rank_2021_columns, merchants_rank_2022_columns,
        gmv_rank_2019_columns, gmv_rank_2020_columns, gmv_rank_2021_columns, gmv_rank_2022_columns,
        avg_traffic_2019_columns, avg_traffic_2020_columns, avg_traffic_2021_columns, avg_traffic_2022_columns
    ]
    columns_group_names = [
        'orders_rank_2019', 'orders_rank_2020', 'orders_rank_2021', 'orders_rank_2022',
        'merchants_rank_2019', 'merchants_rank_2020', 'merchants_rank_2021', 'merchants_rank_2022',
        'gmv_rank_2019', 'gmv_rank_2020', 'gmv_rank_2021', 'gmv_rank_2022',
        'avg_traffic_2019', 'avg_traffic_2020', 'avg_traffic_2021', 'avg_traffic_2022',
    ]
    # Aggregate orders rank, merchants rank, gmv rank density and avg traffic on years for every dataset
    for columns_group_name, column_group in zip(columns_group_names, column_groups):
        for df in [df_indcom_cbsas, df_indcom_cities, df_indcom_counties, df_indcom_states]:
            df[f'{columns_group_name}_mean'] = df[column_group].mean(axis=1)
            df[f'{columns_group_name}_std'] = df[column_group].std(axis=1)
            df[f'{columns_group_name}_min'] = df[column_group].min(axis=1)
            df[f'{columns_group_name}_max'] = df[column_group].max(axis=1)
            df[f'{columns_group_name}_sum'] = df[column_group].sum(axis=1)

    # Aggregate features on state groups and merge it to state data
    df_indcom_cities.drop(columns=['city_id', 'city'], inplace=True)
    df_indcom_cities_aggregations = df_indcom_cities.groupby('state_abbrev').agg(['mean', 'std', 'min', 'max', 'sum'])
    df_indcom_cities_aggregations.columns = 'state_' + df_indcom_cities_aggregations.columns.map('_'.join).str.strip('_')
    df_indcom_cities_aggregations.reset_index(inplace=True)

    for column in df_indcom_cities_aggregations.columns:
        if df_indcom_cities_aggregations[column].dtype == 'float64':
            df_indcom_cities_aggregations[column] = df_indcom_cities_aggregations[column].astype(np.float32)

    for column in df_indcom_states.columns:
        if df_indcom_states[column].dtype == 'float64':
            df_indcom_states[column] = df_indcom_states[column].astype(np.float32)

    df_indcom_states.columns = [f'state_{column}' for column in df_indcom_states.columns]
    df_indcom_states.rename(columns={'state_state_abbrev': 'state_abbrev', 'state_fips': 'fips'}, inplace=True)

    df_indcom_states = df_indcom_states.merge(df_indcom_cities_aggregations, on='state_abbrev', how='left')
    df_indcom_states = df_indcom_states.drop(columns=['state_state_name', 'state_abbrev']).sort_values(by='fips', ascending=True).reset_index(drop=True)
    df_indcom_states.to_parquet(external_processed_dataset_directory / 'state_industry_commerce.parquet')
    settings.logger.info(f'state_industry_commerce.parquet is saved to {external_processed_dataset_directory}')

    for column in df_indcom_counties.columns:
        if df_indcom_counties[column].dtype == 'float64':
            df_indcom_counties[column] = df_indcom_counties[column].astype(np.float32)

    # Drop counties that doesn't exist in training and test sets
    df_indcom_counties = df_indcom_counties.loc[df_indcom_counties['cfips'].isin(df_train['cfips'].unique())]
    df_indcom_counties = df_indcom_counties.sort_values(by='cfips', ascending=True).reset_index(drop=True)
    df_indcom_counties['cfips'] = df_indcom_counties['cfips'].astype('int')
    df_indcom_counties['total_pop_20'] = df_indcom_counties['total_pop_20'].astype('int')
    df_indcom_counties.drop(columns=['county', 'state', 'groupflag'], inplace=True)
    df_indcom_counties.columns = [f'county_{column}' for column in df_indcom_counties.columns]
    df_indcom_counties.rename(columns={'county_cfips': 'cfips'}, inplace=True)
    df_indcom_counties.to_parquet(external_processed_dataset_directory / 'county_industry_commerce.parquet')
    settings.logger.info(f'county_industry_commerce.parquet is saved to {external_processed_dataset_directory}')
