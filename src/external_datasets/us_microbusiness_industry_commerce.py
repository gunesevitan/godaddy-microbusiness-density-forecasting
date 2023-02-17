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

    datetime_idx = pd.date_range(start='2019-08-01', end='2022-06-01', freq='MS')

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

    # Create cartesian index from unique cities and months
    unique_cities = df_indcom_cities['city_id'].unique().tolist()
    cities_idx = pd.MultiIndex.from_product([unique_cities, datetime_idx], names=['city_id', 'first_day_of_month'])
    df_indcom_cities_reindexed = pd.DataFrame(index=cities_idx).reset_index()

    # Melt and merge orders_rank values to re-indexed data
    df_indcom_cities_orders_rank = pd.melt(
        df_indcom_cities[['city_id'] + orders_rank_2019_columns + orders_rank_2020_columns + orders_rank_2021_columns + orders_rank_2022_columns],
        id_vars='city_id'
    ).rename(columns={'value': 'orders_rank'})
    df_indcom_cities_orders_rank['first_day_of_month'] = pd.to_datetime(df_indcom_cities_orders_rank['variable'].apply(lambda x: f'{x[12:-2]}-20{x[-2:]}'))
    df_indcom_cities_reindexed = df_indcom_cities_reindexed.merge(
        df_indcom_cities_orders_rank.drop(columns=['variable']),
        on=['city_id', 'first_day_of_month'],
        how='left'
    )
    del df_indcom_cities_orders_rank

    # Melt and merge merchants_rank values to re-indexed data
    df_indcom_cities_merchants_rank = pd.melt(
        df_indcom_cities[['city_id'] + merchants_rank_2019_columns + merchants_rank_2020_columns + merchants_rank_2021_columns + merchants_rank_2022_columns],
        id_vars='city_id'
    ).rename(columns={'value': 'merchants_rank'})
    df_indcom_cities_merchants_rank['first_day_of_month'] = pd.to_datetime(df_indcom_cities_merchants_rank['variable'].apply(lambda x: f'{x[15:-2]}-20{x[-2:]}'))
    df_indcom_cities_reindexed = df_indcom_cities_reindexed.merge(
        df_indcom_cities_merchants_rank.drop(columns=['variable']),
        on=['city_id', 'first_day_of_month'],
        how='left'
    )
    del df_indcom_cities_merchants_rank

    # Melt and merge gmv_rank values to re-indexed data
    df_indcom_cities_gmv_rank = pd.melt(
        df_indcom_cities[['city_id'] + gmv_rank_2019_columns + gmv_rank_2020_columns + gmv_rank_2021_columns + gmv_rank_2022_columns],
        id_vars='city_id'
    ).rename(columns={'value': 'gmv_rank'})
    df_indcom_cities_gmv_rank['first_day_of_month'] = pd.to_datetime(df_indcom_cities_gmv_rank['variable'].apply(lambda x: f'{x[9:-2]}-20{x[-2:]}'))
    df_indcom_cities_reindexed = df_indcom_cities_reindexed.merge(
        df_indcom_cities_gmv_rank.drop(columns=['variable']),
        on=['city_id', 'first_day_of_month'],
        how='left'
    )
    del df_indcom_cities_gmv_rank

    # Melt and merge avg_traffic values to re-indexed data
    df_indcom_cities_avg_traffic = pd.melt(
        df_indcom_cities[['city_id'] + avg_traffic_2019_columns + avg_traffic_2020_columns + avg_traffic_2021_columns + avg_traffic_2022_columns],
        id_vars='city_id'
    ).rename(columns={'value': 'avg_traffic'})
    df_indcom_cities_avg_traffic['first_day_of_month'] = pd.to_datetime(df_indcom_cities_avg_traffic['variable'].apply(lambda x: f'{x[12:-2]}-20{x[-2:]}'))
    df_indcom_cities_reindexed = df_indcom_cities_reindexed.merge(
        df_indcom_cities_avg_traffic.drop(columns=['variable']),
        on=['city_id', 'first_day_of_month'],
        how='left'
    )
    del df_indcom_cities_avg_traffic

    # Merge remaining columns and create aggregations
    df_indcom_cities_reindexed = df_indcom_cities_reindexed.merge(df_indcom_cities[[
        'city_id', 'state_abbrev'
    ]], on=['city_id'], how='left')
    df_indcom_cities_reindexed_state_aggregations = df_indcom_cities_reindexed.groupby(['first_day_of_month', 'state_abbrev']).agg({
        'orders_rank': ['mean', 'std', 'min', 'max', 'sum'],
        'merchants_rank': ['mean', 'std', 'min', 'max', 'sum'],
        'gmv_rank': ['mean', 'std', 'min', 'max', 'sum'],
        'avg_traffic': ['mean', 'std', 'min', 'max', 'sum']
    })
    del df_indcom_cities_reindexed
    df_indcom_cities_reindexed_state_aggregations.columns = 'state_' + df_indcom_cities_reindexed_state_aggregations.columns.map('_'.join).str.strip('_')
    df_indcom_cities_reindexed_state_aggregations.reset_index(inplace=True)

    # Create cartesian index from unique states and months
    unique_states = df_indcom_states['state_abbrev'].unique().tolist()
    states_idx = pd.MultiIndex.from_product([unique_states, datetime_idx], names=['state_abbrev', 'first_day_of_month'])
    df_indcom_states_reindexed = pd.DataFrame(index=states_idx).reset_index()

    # Melt and merge orders_rank values to re-indexed data
    df_indcom_states_orders_rank = pd.melt(
        df_indcom_states[['state_abbrev'] + orders_rank_2019_columns + orders_rank_2020_columns + orders_rank_2021_columns + orders_rank_2022_columns],
        id_vars='state_abbrev'
    ).rename(columns={'value': 'orders_rank'})
    df_indcom_states_orders_rank['first_day_of_month'] = pd.to_datetime(df_indcom_states_orders_rank['variable'].apply(lambda x: f'{x[12:-2]}-20{x[-2:]}'))
    df_indcom_states_reindexed = df_indcom_states_reindexed.merge(
        df_indcom_states_orders_rank.drop(columns=['variable']),
        on=['state_abbrev', 'first_day_of_month'],
        how='left'
    )
    del df_indcom_states_orders_rank

    # Melt and merge merchants_rank values to re-indexed data
    df_indcom_states_merchants_rank = pd.melt(
        df_indcom_states[['state_abbrev'] + merchants_rank_2019_columns + merchants_rank_2020_columns + merchants_rank_2021_columns + merchants_rank_2022_columns],
        id_vars='state_abbrev'
    ).rename(columns={'value': 'merchants_rank'})
    df_indcom_states_merchants_rank['first_day_of_month'] = pd.to_datetime(df_indcom_states_merchants_rank['variable'].apply(lambda x: f'{x[15:-2]}-20{x[-2:]}'))
    df_indcom_states_reindexed = df_indcom_states_reindexed.merge(
        df_indcom_states_merchants_rank.drop(columns=['variable']),
        on=['state_abbrev', 'first_day_of_month'],
        how='left'
    )
    del df_indcom_states_merchants_rank

    # Melt and merge gmv_rank values to re-indexed data
    df_indcom_states_gmv_rank = pd.melt(
        df_indcom_states[['state_abbrev'] + gmv_rank_2019_columns + gmv_rank_2020_columns + gmv_rank_2021_columns + gmv_rank_2022_columns],
        id_vars='state_abbrev'
    ).rename(columns={'value': 'gmv_rank'})
    df_indcom_states_gmv_rank['first_day_of_month'] = pd.to_datetime(df_indcom_states_gmv_rank['variable'].apply(lambda x: f'{x[9:-2]}-20{x[-2:]}'))
    df_indcom_states_reindexed = df_indcom_states_reindexed.merge(
        df_indcom_states_gmv_rank.drop(columns=['variable']),
        on=['state_abbrev', 'first_day_of_month'],
        how='left'
    )
    del df_indcom_states_gmv_rank

    # Melt and merge avg_traffic values to re-indexed data
    df_indcom_states_avg_traffic = pd.melt(
        df_indcom_states[['state_abbrev'] + avg_traffic_2019_columns + avg_traffic_2020_columns + avg_traffic_2021_columns + avg_traffic_2022_columns],
        id_vars='state_abbrev'
    ).rename(columns={'value': 'avg_traffic'})
    df_indcom_states_avg_traffic['first_day_of_month'] = pd.to_datetime(df_indcom_states_avg_traffic['variable'].apply(lambda x: f'{x[12:-2]}-20{x[-2:]}'))
    df_indcom_states_reindexed = df_indcom_states_reindexed.merge(
        df_indcom_states_avg_traffic.drop(columns=['variable']),
        on=['state_abbrev', 'first_day_of_month'],
        how='left'
    )
    del df_indcom_states_avg_traffic

    # Merge remaining columns
    df_indcom_states_reindexed = df_indcom_states_reindexed.merge(df_indcom_states[[
        'state_abbrev', 'fips', 'total_pop_20'
    ]].rename(columns={'total_pop_20': 'state_total_pop_20'})).merge(
        df_indcom_cities_reindexed_state_aggregations,
        on=['state_abbrev', 'first_day_of_month'],
        how='left'
    ).drop(columns=['state_abbrev']).rename(columns={
        'orders_rank': 'state_orders_rank',
        'merchants_rank': 'state_merchants_rank',
        'gmv_rank': 'state_gmv_rank',
        'avg_traffic': 'state_avg_traffic',
    })
    df_indcom_states_reindexed = df_indcom_states_reindexed[sorted(df_indcom_states_reindexed.columns)]
    del df_indcom_states

    for column in df_indcom_states_reindexed.columns:
        if df_indcom_states_reindexed[column].dtype == 'float64':
            df_indcom_states_reindexed[column] = df_indcom_states_reindexed[column].astype(np.float32)

    df_indcom_states_reindexed.to_parquet(external_processed_dataset_directory / 'state_industry_commerce.parquet')
    settings.logger.info(f'state_industry_commerce.parquet is saved to {external_processed_dataset_directory}')

    for column in df_indcom_counties.columns:
        if df_indcom_counties[column].dtype == 'float64':
            df_indcom_counties[column] = df_indcom_counties[column].astype(np.float32)

    # Drop counties that doesn't exist in training and test sets
    df_indcom_counties = df_indcom_counties.loc[df_indcom_counties['cfips'].isin(df_train['cfips'].unique())]
    df_indcom_counties = df_indcom_counties.sort_values(by='cfips', ascending=True).reset_index(drop=True)
    df_indcom_counties['cfips'] = df_indcom_counties['cfips'].astype('int')
    df_indcom_counties['total_pop_20'] = df_indcom_counties['total_pop_20'].astype('int')
    df_indcom_counties.drop(columns=['county', 'state'], inplace=True)

    # Create cartesian index from unique states and months
    unique_counties = df_indcom_counties['cfips'].unique().tolist()
    counties_idx = pd.MultiIndex.from_product([unique_counties, datetime_idx], names=['cfips', 'first_day_of_month'])
    df_indcom_counties_reindexed = pd.DataFrame(index=counties_idx).reset_index()

    # Melt and merge orders_rank values to re-indexed data
    df_indcom_counties_orders_rank = pd.melt(
        df_indcom_counties[['cfips'] + orders_rank_2019_columns + orders_rank_2020_columns + orders_rank_2021_columns + orders_rank_2022_columns],
        id_vars='cfips'
    ).rename(columns={'value': 'orders_rank'})
    df_indcom_counties_orders_rank['first_day_of_month'] = pd.to_datetime(df_indcom_counties_orders_rank['variable'].apply(lambda x: f'{x[12:-2]}-20{x[-2:]}'))
    df_indcom_counties_reindexed = df_indcom_counties_reindexed.merge(
        df_indcom_counties_orders_rank.drop(columns=['variable']),
        on=['cfips', 'first_day_of_month'],
        how='left'
    )
    del df_indcom_counties_orders_rank

    # Melt and merge merchants_rank values to re-indexed data
    df_indcom_counties_merchants_rank = pd.melt(
        df_indcom_counties[['cfips'] + merchants_rank_2019_columns + merchants_rank_2020_columns + merchants_rank_2021_columns + merchants_rank_2022_columns],
        id_vars='cfips'
    ).rename(columns={'value': 'merchants_rank'})
    df_indcom_counties_merchants_rank['first_day_of_month'] = pd.to_datetime(df_indcom_counties_merchants_rank['variable'].apply(lambda x: f'{x[15:-2]}-20{x[-2:]}'))
    df_indcom_counties_reindexed = df_indcom_counties_reindexed.merge(
        df_indcom_counties_merchants_rank.drop(columns=['variable']),
        on=['cfips', 'first_day_of_month'],
        how='left'
    )
    del df_indcom_counties_merchants_rank

    # Melt and merge gmv_rank values to re-indexed data
    df_indcom_counties_gmv_rank = pd.melt(
        df_indcom_counties[['cfips'] + gmv_rank_2019_columns + gmv_rank_2020_columns + gmv_rank_2021_columns + gmv_rank_2022_columns],
        id_vars='cfips'
    ).rename(columns={'value': 'gmv_rank'})
    df_indcom_counties_gmv_rank['first_day_of_month'] = pd.to_datetime(df_indcom_counties_gmv_rank['variable'].apply(lambda x: f'{x[9:-2]}-20{x[-2:]}'))
    df_indcom_counties_reindexed = df_indcom_counties_reindexed.merge(
        df_indcom_counties_gmv_rank.drop(columns=['variable']),
        on=['cfips', 'first_day_of_month'],
        how='left'
    )
    del df_indcom_counties_gmv_rank

    # Melt and merge avg_traffic values to re-indexed data
    df_indcom_counties_avg_traffic = pd.melt(
        df_indcom_counties[['cfips'] + avg_traffic_2019_columns + avg_traffic_2020_columns + avg_traffic_2021_columns + avg_traffic_2022_columns],
        id_vars='cfips'
    ).rename(columns={'value': 'avg_traffic'})
    df_indcom_counties_avg_traffic['first_day_of_month'] = pd.to_datetime(df_indcom_counties_avg_traffic['variable'].apply(lambda x: f'{x[12:-2]}-20{x[-2:]}'))
    df_indcom_counties_reindexed = df_indcom_counties_reindexed.merge(
        df_indcom_counties_avg_traffic.drop(columns=['variable']),
        on=['cfips', 'first_day_of_month'],
        how='left'
    )
    del df_indcom_counties_avg_traffic

    df_indcom_counties_reindexed = df_indcom_counties_reindexed.merge(df_indcom_counties[[
        'cfips', 'total_pop_20'
    ]].rename(columns={'total_pop_20': 'county_total_pop_20'}))
    df_indcom_counties_reindexed.rename(columns={
        'orders_rank': 'county_orders_rank',
        'merchants_rank': 'county_merchants_rank',
        'gmv_rank': 'county_gmv_rank',
        'avg_traffic': 'county_avg_traffic',
    }, inplace=True)

    df_indcom_counties_reindexed.to_parquet(external_processed_dataset_directory / 'county_industry_commerce.parquet')
    settings.logger.info(f'county_industry_commerce.parquet is saved to {external_processed_dataset_directory}')
