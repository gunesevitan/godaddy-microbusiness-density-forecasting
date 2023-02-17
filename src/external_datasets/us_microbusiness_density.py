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

    datetime_idx = pd.date_range(start='2019-08-01', end='2022-06-01', freq='MS')

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

    # Create cartesian index from unique cities and months
    unique_cities = df_md_cities['city_id'].unique().tolist()
    cities_idx = pd.MultiIndex.from_product([unique_cities, datetime_idx], names=['city_id', 'first_day_of_month'])
    df_md_cities_reindexed = pd.DataFrame(index=cities_idx).reset_index()

    # Melt and merge active values to re-indexed data
    df_md_cities_active = pd.melt(df_md_cities[
        ['city_id'] + active_2019_columns + active_2020_columns + active_2021_columns + active_2022_columns
    ], id_vars='city_id').rename(columns={'value': 'active'})
    df_md_cities_active['first_day_of_month'] = pd.to_datetime(df_md_cities_active['variable'].apply(lambda x: f'{x[6:-2]}-20{x[-2:]}'))
    df_md_cities_reindexed = df_md_cities_reindexed.merge(
        df_md_cities_active.drop(columns=['variable']),
        on=['city_id', 'first_day_of_month'],
        how='left'
    )
    del df_md_cities_active

    # Melt and merge md values to re-indexed data
    df_md_cities_md = pd.melt(df_md_cities[
        ['city_id'] + md_2019_columns + md_2020_columns + md_2021_columns + md_2022_columns
    ], id_vars='city_id').rename(columns={'value': 'md'})
    df_md_cities_md['first_day_of_month'] = pd.to_datetime(df_md_cities_md['variable'].apply(lambda x: f'{x[2:-2]}-20{x[-2:]}'))
    df_md_cities_reindexed = df_md_cities_reindexed.merge(
        df_md_cities_md.drop(columns=['variable']),
        on=['city_id', 'first_day_of_month'],
        how='left'
    )
    del df_md_cities_md

    # Merge remaining columns and create aggregations
    df_md_cities_reindexed = df_md_cities_reindexed.merge(df_md_cities[[
        'city_id', 'state_abbrev', 'pop_18over_2020'
    ]].rename(columns={'pop_18over_2020': 'city_pop_18over_2020'}), on=['city_id'], how='left')
    df_md_cities_reindexed['year'] = df_md_cities_reindexed['first_day_of_month'].dt.year
    df_md_cities_reindexed_state_aggregations = df_md_cities_reindexed.groupby(['first_day_of_month', 'state_abbrev']).agg({
        'active': ['mean', 'std', 'min', 'max', 'sum'],
        'md': ['mean', 'std', 'min', 'max', 'sum'],
    })
    del df_md_cities_reindexed
    df_md_cities_reindexed_state_aggregations.columns = 'state_' + df_md_cities_reindexed_state_aggregations.columns.map('_'.join).str.strip('_')
    df_md_cities_reindexed_state_aggregations.reset_index(inplace=True)

    # Create cartesian index from unique states and months
    unique_states = df_md_states['state_abbrev'].unique().tolist()
    states_idx = pd.MultiIndex.from_product([unique_states, datetime_idx], names=['state_abbrev', 'first_day_of_month'])
    df_md_states_reindexed = pd.DataFrame(index=states_idx).reset_index()

    # Melt and merge active values to re-indexed data
    df_md_states_active = pd.melt(df_md_states[
        ['state_abbrev'] + active_2019_columns + active_2020_columns + active_2021_columns + active_2022_columns
    ], id_vars='state_abbrev').rename(columns={'value': 'active'})
    df_md_states_active['first_day_of_month'] = pd.to_datetime(df_md_states_active['variable'].apply(lambda x: f'{x[6:-2]}-20{x[-2:]}'))
    df_md_states_reindexed = df_md_states_reindexed.merge(
        df_md_states_active.drop(columns=['variable']),
        on=['state_abbrev', 'first_day_of_month'],
        how='left'
    )
    del df_md_states_active

    # Melt and merge md values to re-indexed data
    df_md_states_md = pd.melt(df_md_states[
        ['state_abbrev'] + md_2019_columns + md_2020_columns + md_2021_columns + md_2022_columns
    ], id_vars='state_abbrev').rename(columns={'value': 'md'})
    df_md_states_md['first_day_of_month'] = pd.to_datetime(df_md_states_md['variable'].apply(lambda x: f'{x[2:-2]}-20{x[-2:]}'))
    df_md_states_reindexed = df_md_states_reindexed.merge(
        df_md_states_md.drop(columns=['variable']),
        on=['state_abbrev', 'first_day_of_month'],
        how='left'
    )
    del df_md_states_md

    # Merge remaining columns
    df_md_states_reindexed = df_md_states_reindexed.merge(df_md_states[[
        'state_abbrev', 'fips', 'total_pop_20'
    ]].rename(columns={'total_pop_20': 'state_total_pop_20'})).merge(
        df_md_cities_reindexed_state_aggregations,
        on=['state_abbrev', 'first_day_of_month'],
        how='left'
    ).drop(columns=['state_abbrev']).rename(columns={'active': 'state_active', 'md': 'state_md'})
    df_md_states_reindexed = df_md_states_reindexed[sorted(df_md_states_reindexed.columns)]
    del df_md_states

    for column in df_md_states_reindexed.columns:
        if df_md_states_reindexed[column].dtype == 'float64':
            df_md_states_reindexed[column] = df_md_states_reindexed[column].astype(np.float32)

    df_md_states_reindexed.to_parquet(external_processed_dataset_directory / 'state_microbusiness_densities.parquet')
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

    # Create cartesian index from unique states and months
    unique_counties = df_md_counties['cfips'].unique().tolist()
    counties_idx = pd.MultiIndex.from_product([unique_counties, datetime_idx], names=['cfips', 'first_day_of_month'])
    df_md_counties_reindexed = pd.DataFrame(index=counties_idx).reset_index()

    # Melt and merge active values to re-indexed data
    df_md_counties_active = pd.melt(df_md_counties[
        ['cfips'] + active_2019_columns + active_2020_columns + active_2021_columns + active_2022_columns
    ], id_vars='cfips').rename(columns={'value': 'active'})
    df_md_counties_active['first_day_of_month'] = pd.to_datetime(df_md_counties_active['variable'].apply(lambda x: f'{x[6:-2]}-20{x[-2:]}'))
    df_md_counties_reindexed = df_md_counties_reindexed.merge(
        df_md_counties_active.drop(columns=['variable']),
        on=['cfips', 'first_day_of_month'],
        how='left'
    )
    del df_md_counties_active

    # Melt and merge md values to re-indexed data
    df_md_counties_md = pd.melt(df_md_counties[
        ['cfips'] + md_2019_columns + md_2020_columns + md_2021_columns + md_2022_columns
    ], id_vars='cfips').rename(columns={'value': 'md'})
    df_md_counties_md['first_day_of_month'] = pd.to_datetime(df_md_counties_md['variable'].apply(lambda x: f'{x[2:-2]}-20{x[-2:]}'))
    df_md_counties_reindexed = df_md_counties_reindexed.merge(
        df_md_counties_md.drop(columns=['variable']),
        on=['cfips', 'first_day_of_month'],
        how='left'
    )
    del df_md_counties_md

    df_md_counties_reindexed = df_md_counties_reindexed.merge(df_md_counties[[
        'cfips', 'total_pop_20'
    ]].rename(columns={'total_pop_20': 'county_total_pop_20'}))
    df_md_counties_reindexed.rename(columns={'active': 'county_active', 'md': 'county_md'}, inplace=True)

    df_md_counties_reindexed.to_parquet(external_processed_dataset_directory / 'county_microbusiness_densities.parquet')
    settings.logger.info(f'county_microbusiness_densities.parquet is saved to {external_processed_dataset_directory}')
