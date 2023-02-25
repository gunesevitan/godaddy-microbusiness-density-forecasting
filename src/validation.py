import numpy as np
import pandas as pd

import settings


def create_validation_splits(df, validation_folds):

    """
    Create folds as columns on given dataframe

    Parameters
    ----------
    df: pandas.DataFrame
        Dataframe with first_day_of_month column

    validation_folds: dict
        Dictionary of training/validation start/end dates

    Returns
    -------
    df: pandas.DataFrame
        Dataframe with fold columns
    """

    for fold, train_val_dates in validation_folds.items():
        df[fold] = np.nan
        df.loc[(df['first_day_of_month'] >= train_val_dates['train']['start']) & (df['first_day_of_month'] < train_val_dates['train']['end']), fold] = 0
        df.loc[(df['first_day_of_month'] >= train_val_dates['val']['start']) & (df['first_day_of_month'] < train_val_dates['val']['end']), fold] = 1

        if df[fold].isnull().any():
            df[fold] = df[fold].astype(np.float32)
        else:
            df[fold] = df[fold].astype(np.uint8)

    return df


if __name__ == '__main__':

    df_train = pd.read_parquet(settings.DATA / 'train.parquet')
    df_train = df_train.merge(pd.read_parquet(settings.DATA / 'census_starter.parquet'))
    settings.logger.info(f'Training Set Shape: {df_train.shape} - Memory Usage: {df_train.memory_usage().sum() / 1024 ** 2:.2f}')

    df_train = create_validation_splits(
        df=df_train,
        validation_folds={
            # Folds for validating on most frequent year
            'fold1': {
                'train': {
                    'start': '2019-08-01 00:00:00',
                    'end': '2022-01-01 00:00:00'
                },
                'val': {
                    'start': '2022-03-01 00:00:00',
                    'end': '2022-06-01 00:00:00'
                }
            },
            'fold2': {
                'train': {
                    'start': '2019-08-01 00:00:00',
                    'end': '2022-02-01 00:00:00'
                },
                'val': {
                    'start': '2022-04-01 00:00:00',
                    'end': '2022-07-01 00:00:00'
                }
            },
            'fold3': {
                'train': {
                    'start': '2019-08-01 00:00:00',
                    'end': '2022-03-01 00:00:00'
                },
                'val': {
                    'start': '2022-05-01 00:00:00',
                    'end': '2022-08-01 00:00:00'
                }
            },
            'fold4': {
                'train': {
                    'start': '2019-08-01 00:00:00',
                    'end': '2022-04-01 00:00:00'
                },
                'val': {
                    'start': '2022-06-01 00:00:00',
                    'end': '2022-09-01 00:00:00'
                }
            },
            'fold5': {
                'train': {
                    'start': '2019-08-01 00:00:00',
                    'end': '2022-05-01 00:00:00'
                },
                'val': {
                    'start': '2022-07-01 00:00:00',
                    'end': '2022-10-01 00:00:00'
                }
            },
            'fold6': {
                'train': {
                    'start': '2019-08-01 00:00:00',
                    'end': '2022-06-01 00:00:00'
                },
                'val': {
                    'start': '2022-08-01 00:00:00',
                    'end': '2022-11-01 00:00:00'
                }
            },
            'fold7': {
                'train': {
                    'start': '2019-08-01 00:00:00',
                    'end': '2022-07-01 00:00:00'
                },
                'val': {
                    'start': '2022-09-01 00:00:00',
                    'end': '2022-12-01 00:00:00'
                }
            },
            'fold8': {
                'train': {
                    'start': '2019-08-01 00:00:00',
                    'end': '2022-08-01 00:00:00'
                },
                'val': {
                    'start': '2022-10-01 00:00:00',
                    'end': '2023-01-01 00:00:00'
                }
            },
            # Folds for validating new year transitions
            'fold9': {
                'train': {
                    'start': '2019-08-01 00:00:00',
                    'end': '2020-01-01 00:00:00'
                },
                'val': {
                    'start': '2020-03-01 00:00:00',
                    'end': '2020-06-01 00:00:00'
                }
            },
            'fold10': {
                'train': {
                    'start': '2019-08-01 00:00:00',
                    'end': '2021-01-01 00:00:00'
                },
                'val': {
                    'start': '2021-03-01 00:00:00',
                    'end': '2021-06-01 00:00:00'
                }
            },
            'fold11': {
                'train': {
                    'start': '2019-08-01 00:00:00',
                    'end': '2022-01-01 00:00:00'
                },
                'val': {
                    'start': '2022-03-01 00:00:00',
                    'end': '2022-06-01 00:00:00'
                }
            },
        }
    )

    fold_columns = [column for column in df_train.columns if column.startswith('fold')]
    for fold_column in fold_columns:
        settings.logger.info(
            f'''
            {fold_column}
            Training: {df_train.loc[df_train[fold_column] == 0, 'first_day_of_month'].min()} - {df_train.loc[df_train[fold_column] == 0, 'first_day_of_month'].max()}
            {df_train.loc[df_train[fold_column] == 0].shape[0]} Samples
            Validation: {df_train.loc[df_train[fold_column] == 1, 'first_day_of_month'].min()} - {df_train.loc[df_train[fold_column] == 1, 'first_day_of_month'].max()}
            {df_train.loc[df_train[fold_column] == 1].shape[0]} Samples 
            '''
        )

    df_train[['row_id'] + fold_columns].to_csv(settings.DATA / 'folds.csv', index=False)
    settings.logger.info(f'folds.csv is saved to {settings.DATA}')
