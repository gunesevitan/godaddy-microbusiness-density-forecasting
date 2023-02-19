import sys
import pathlib
import numpy as np
import pandas as pd

sys.path.append('..')
import settings


if __name__ == '__main__':

    external_raw_dataset_directory = pathlib.Path(settings.DATA / 'external' / 'raw' / 'VF_GB_bundle_Q121')
    external_processed_dataset_directory = pathlib.Path(settings.DATA / 'external' / 'processed')

    df_gb = pd.read_csv(external_raw_dataset_directory / 'VF_GB_constituencies_Q121.csv')
    settings.logger.info(f'VF_GB_constituencies_Q121 Shape: {df_gb.shape} - Memory Usage: {df_gb.memory_usage().sum() / 1024 ** 2:.2f}')

    df_train = pd.read_parquet(settings.DATA / 'train.parquet')
    settings.logger.info(f'train Shape: {df_train.shape} - Memory Usage: {df_train.memory_usage().sum() / 1024 ** 2:.2f}')

    df_test = pd.read_parquet(settings.DATA / 'test.parquet')
    settings.logger.info(f'test Shape: {df_test.shape} - Memory Usage: {df_test.memory_usage().sum() / 1024 ** 2:.2f}')

    datetime_idx = pd.date_range(start='2020-04-01', end='2021-03-01', freq='MS')

    # Select columns of each year separately
    counts_2020_columns = [column for column in df_gb.columns if column.startswith('counts') and column.endswith('20')]
    counts_2021_columns = [column for column in df_gb.columns if column.startswith('counts') and column.endswith('21')]
    md_2020_columns = [column for column in df_gb.columns if column.startswith('md') and column.endswith('20')]
    md_2021_columns = [column for column in df_gb.columns if column.startswith('md') and column.endswith('21')]

    # Create cartesian index from unique cities and months
    unique_constits = df_gb['constit'].unique().tolist()
    gb_idx = pd.MultiIndex.from_product([unique_constits, datetime_idx], names=['constit', 'first_day_of_month'])
    df_gb_reindexed = pd.DataFrame(index=gb_idx).reset_index()

    # Melt and merge counts values to re-indexed data
    df_gb_counts = pd.melt(
        df_gb[['constit'] + counts_2020_columns + counts_2021_columns],
        id_vars='constit'
    ).rename(columns={'value': 'counts'})
    df_gb_counts['first_day_of_month'] = pd.to_datetime(df_gb_counts['variable'].apply(lambda x: f'{x[7:-2]}-20{x[-2:]}'))
    df_gb_reindexed = df_gb_reindexed.merge(
        df_gb_counts.drop(columns=['variable']),
        on=['constit', 'first_day_of_month'],
        how='left'
    )
    del df_gb_counts

    # Melt and merge md values to re-indexed data
    df_gb_md = pd.melt(
        df_gb[['constit'] + md_2020_columns + md_2021_columns],
        id_vars='constit'
    ).rename(columns={'value': 'md'})
    df_gb_md['first_day_of_month'] = pd.to_datetime(df_gb_md['variable'].apply(lambda x: f'{x[3:-2]}-20{x[-2:]}'))
    df_gb_reindexed = df_gb_reindexed.merge(
        df_gb_md.drop(columns=['variable']),
        on=['constit', 'first_day_of_month'],
        how='left'
    )
    del df_gb_md

    df_gb_reindexed.to_parquet(external_processed_dataset_directory / 'great_britain_microbusiness_density.parquet')
    settings.logger.info(f'great_britain_microbusiness_density.parquet is saved to {external_processed_dataset_directory}')
