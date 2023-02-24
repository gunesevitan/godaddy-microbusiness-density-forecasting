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

    pct_bb_columns = [column for column in df_census_starter.columns if column.startswith('pct_bb')]
    pct_college_columns = [column for column in df_census_starter.columns if column.startswith('pct_college')]
    pct_foreign_born_columns = [column for column in df_census_starter.columns if column.startswith('pct_foreign_born')]
    pct_it_workers_columns = [column for column in df_census_starter.columns if column.startswith('pct_it_workers')]
    median_hh_inc_columns = [column for column in df_census_starter.columns if column.startswith('median_hh_inc')]

    # Create cartesian index from unique cities and years
    datetime_idx = pd.date_range(start='2017-01-01', end='2021-01-01', freq='YS')
    unique_counties = df_census_starter['cfips'].unique().tolist()
    census_idx = pd.MultiIndex.from_product([unique_counties, datetime_idx], names=['cfips', 'first_day_of_year'])
    df_census_reindexed = pd.DataFrame(index=census_idx).reset_index()

    # Melt and merge pct_bb values to re-indexed data
    df_pct_bb = pd.melt(
        df_census_starter[['cfips'] + pct_bb_columns],
        id_vars='cfips'
    ).rename(columns={'value': 'pct_bb'})
    df_pct_bb['first_day_of_year'] = pd.to_datetime(df_pct_bb['variable'].apply(lambda x: f'{x[7:]}-01-01'))
    df_census_reindexed = df_census_reindexed.merge(
        df_pct_bb.drop(columns=['variable']),
        on=['cfips', 'first_day_of_year'],
        how='left'
    )
    del df_pct_bb

    # Melt and merge pct_college values to re-indexed data
    df_pct_college = pd.melt(
        df_census_starter[['cfips'] + pct_college_columns],
        id_vars='cfips'
    ).rename(columns={'value': 'pct_college'})
    df_pct_college['first_day_of_year'] = pd.to_datetime(df_pct_college['variable'].apply(lambda x: f'{x[12:]}-01-01'))
    df_census_reindexed = df_census_reindexed.merge(
        df_pct_college.drop(columns=['variable']),
        on=['cfips', 'first_day_of_year'],
        how='left'
    )
    del df_pct_college

    # Melt and merge pct_foreign_born values to re-indexed data
    df_pct_foreign_born = pd.melt(
        df_census_starter[['cfips'] + pct_foreign_born_columns],
        id_vars='cfips'
    ).rename(columns={'value': 'pct_foreign_born'})
    df_pct_foreign_born['first_day_of_year'] = pd.to_datetime(df_pct_foreign_born['variable'].apply(lambda x: f'{x[17:]}-01-01'))
    df_census_reindexed = df_census_reindexed.merge(
        df_pct_foreign_born.drop(columns=['variable']),
        on=['cfips', 'first_day_of_year'],
        how='left'
    )
    del df_pct_foreign_born

    # Melt and merge pct_it_workers values to re-indexed data
    df_pct_it_workers = pd.melt(
        df_census_starter[['cfips'] + pct_it_workers_columns],
        id_vars='cfips'
    ).rename(columns={'value': 'pct_it_workers'})
    df_pct_it_workers['first_day_of_year'] = pd.to_datetime(df_pct_it_workers['variable'].apply(lambda x: f'{x[15:]}-01-01'))
    df_census_reindexed = df_census_reindexed.merge(
        df_pct_it_workers.drop(columns=['variable']),
        on=['cfips', 'first_day_of_year'],
        how='left'
    )
    del df_pct_it_workers

    # Melt and merge median_hh_inc values to re-indexed data
    df_median_hh_inc = pd.melt(
        df_census_starter[['cfips'] + median_hh_inc_columns],
        id_vars='cfips'
    ).rename(columns={'value': 'median_hh_inc'})
    df_median_hh_inc['first_day_of_year'] = pd.to_datetime(df_median_hh_inc['variable'].apply(lambda x: f'{x[14:]}-01-01'))
    df_census_reindexed = df_census_reindexed.merge(
        df_median_hh_inc.drop(columns=['variable']),
        on=['cfips', 'first_day_of_year'],
        how='left'
    )
    del df_median_hh_inc

    df_train = pd.concat((
        df_train,
        df_revealed_test
    ), axis=0, ignore_index=True)

    for df in [df_train, df_test]:

        df['cfips'] = df['cfips'].astype(np.uint32)
        if 'active' in df.columns:
            df['active'] = df['active'].astype(np.uint32)

        df['first_day_of_month'] = pd.to_datetime(df['first_day_of_month'])

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
        'first_day_of_month', 'microbusiness_density', 'active'
    ]

    df_train[columns].to_parquet(settings.DATA / 'train.parquet')
    settings.logger.info(f'train.parquet is saved to {settings.DATA}')

    df_test[columns[:-2]].to_parquet(settings.DATA / 'test.parquet')
    settings.logger.info(f'test.parquet is saved to {settings.DATA}')

    df_census_reindexed.to_parquet(settings.DATA / 'census.parquet')
    settings.logger.info(f'census.parquet is saved to {settings.DATA}')
