import sys
import argparse
from pathlib import Path
import yaml
import json
from tqdm import tqdm
import numpy as np
import pandas as pd

sys.path.append('..')
import settings
import metrics
import visualization


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('model_directory', type=str)
    parser.add_argument('mode', type=str)
    args = parser.parse_args()

    model_directory = Path(settings.MODELS / args.model_directory)
    config = yaml.load(open(model_directory / 'config.yaml'), Loader=yaml.FullLoader)

    df_train = pd.read_parquet(settings.DATA / config['dataset']['train'])
    df_train = df_train.merge(pd.read_csv(settings.DATA / 'folds.csv'), on='row_id', how='left')
    settings.logger.info(f'{config["dataset"]["train"]} Shape: {df_train.shape} - Memory Usage: {df_train.memory_usage().sum() / 1024 ** 2:.2f} MB')

    df_test = pd.read_parquet(settings.DATA / config['dataset']['test'])
    settings.logger.info(f'{config["dataset"]["test"]} Shape: {df_test.shape} - Memory Usage: {df_test.memory_usage().sum() / 1024 ** 2:.2f} MB')

    target = config['training']['target']
    folds = config['training']['folds']

    if args.mode == 'validation':

        settings.logger.info('Running last value model in validation mode')

        last_value_scores = []
        last_value_adjusted_scores = []
        last_value_adjusted_weighted_scores = []

        for fold in folds:

            train_idx, val_idx = df_train.loc[df_train[fold] == 0].index, df_train.loc[df_train[fold] == 1].index
            if len(val_idx) == 0:
                val_idx = train_idx

            settings.logger.info(
                f'''
                {fold}
                Training: ({len(train_idx)}) - [{df_train.loc[train_idx, 'first_day_of_month'].min()} - {df_train.loc[train_idx, 'first_day_of_month'].max()})
                Validation: ({len(val_idx)}) - [{df_train.loc[val_idx, 'first_day_of_month'].min()} - {df_train.loc[val_idx, 'first_day_of_month'].max()})
                '''
            )

            df_train.loc[val_idx, f'{fold}_last_value_predictions'] = df_train.loc[val_idx, 'cfips'].map(df_train.loc[train_idx].groupby('cfips')[target].last())
            last_value_val_scores = metrics.regression_scores(
                y_true=df_train.loc[val_idx, target],
                y_pred=df_train.loc[val_idx, f'{fold}_last_value_predictions'],
            )
            settings.logger.info(f'{fold} - Last Value Validation Scores: {json.dumps(last_value_val_scores, indent=2)}')
            last_value_scores.append(last_value_val_scores)

            val_min_year = df_train.loc[val_idx, 'first_day_of_month'].dt.year.min()
            train_max_year = df_train.loc[train_idx, 'first_day_of_month'].dt.year.max()

            if val_min_year > train_max_year:
                # Adjustment is applied because new year transition occurs
                df_acs_5y = pd.read_parquet(settings.DATA / 'external' / 'processed' / 'acs_5y.parquet')
                df_acs_5y['year'] = df_acs_5y['first_day_of_year'].dt.year + 2
                df_train['year'] = df_train['first_day_of_month'].dt.year
                df_train = df_train.merge(df_acs_5y, on=['cfips', 'year'], how='left')
                if val_min_year == 2020:
                    adjustment_ratios = df_acs_5y.loc[df_acs_5y['first_day_of_year'] == '2017-01-01'].set_index('cfips')['S0101_C01_026E'] /\
                                        df_acs_5y.loc[df_acs_5y['first_day_of_year'] == '2018-01-01'].set_index('cfips')['S0101_C01_026E']
                elif val_min_year == 2021:
                    adjustment_ratios = df_acs_5y.loc[df_acs_5y['first_day_of_year'] == '2018-01-01'].set_index('cfips')['S0101_C01_026E'] /\
                                        df_acs_5y.loc[df_acs_5y['first_day_of_year'] == '2019-01-01'].set_index('cfips')['S0101_C01_026E']
                elif val_min_year == 2022:
                    adjustment_ratios = df_acs_5y.loc[df_acs_5y['first_day_of_year'] == '2019-01-01'].set_index('cfips')['S0101_C01_026E'] /\
                                        df_acs_5y.loc[df_acs_5y['first_day_of_year'] == '2020-01-01'].set_index('cfips')['S0101_C01_026E']
                else:
                    adjustment_ratios = None

                if adjustment_ratios is not None:
                    df_train['adjustment_ratio'] = df_train['cfips'].map(adjustment_ratios)
                    df_train.loc[val_idx, f'{fold}_last_value_adjusted_predictions'] = df_train.loc[val_idx, f'{fold}_last_value_predictions'] *\
                                                                                       df_train.loc[val_idx, 'adjustment_ratio']
            else:
                df_train.loc[val_idx, f'{fold}_last_value_adjusted_predictions'] = df_train.loc[val_idx, f'{fold}_last_value_predictions']

            last_value_adjusted_val_scores = metrics.regression_scores(
                y_true=df_train.loc[val_idx, target],
                y_pred=df_train.loc[val_idx, f'{fold}_last_value_adjusted_predictions'],
            )
            settings.logger.info(f'{fold} - Last Value Adjusted Validation Scores: {json.dumps(last_value_adjusted_val_scores, indent=2)}')
            last_value_adjusted_scores.append(last_value_adjusted_val_scores)

            for cfips, df_cfips_group in tqdm(df_train.groupby('cfips'), total=df_train['cfips'].nunique()):

                n = 5
                last_values = df_cfips_group.loc[df_cfips_group[fold] == 0, target].values[-n:]
                last_values_diff = np.diff(last_values)

                if np.all(last_values_diff > 0):
                    weights = np.array([1.010, 1.015, 1.015])
                elif np.all(last_values_diff < 0):
                    weights = np.array([0.998, 0.994, 0.990])
                else:
                    weights = np.array([1.010, 1.010, 1.010])

                df_train.loc[df_cfips_group.loc[df_cfips_group[fold] == 1].index, f'{fold}_last_value_adjusted_weighted_predictions'] = df_cfips_group.loc[
                    df_cfips_group[fold] == 1, f'{fold}_last_value_adjusted_predictions'
                ] * weights

            last_value_adjusted_weighted_val_scores = metrics.regression_scores(
                y_true=df_train.loc[val_idx, target],
                y_pred=df_train.loc[val_idx, f'{fold}_last_value_adjusted_weighted_predictions'],
            )
            settings.logger.info(f'{fold} - Last Value Adjusted Weighted Validation Scores: {json.dumps(last_value_adjusted_weighted_val_scores, indent=2)}')
            last_value_adjusted_weighted_scores.append(last_value_adjusted_weighted_val_scores)

            if config['persistence']['visualize_val_predictions']:

                val_predictions_visualizations_directory = model_directory / 'val_predictions'
                val_predictions_visualizations_directory.mkdir(parents=True, exist_ok=True)

                for cfips, df_cfips_group in tqdm(df_train.groupby('cfips'), total=df_train['cfips'].nunique()):
                    df_cfips_group = df_cfips_group.sort_values(by='first_day_of_month', ascending=True)
                    visualization.visualize_timeseries_target(
                        df=df_cfips_group,
                        datetime='first_day_of_month',
                        target='microbusiness_density',
                        cfips=cfips,
                        start=df_cfips_group['first_day_of_month'].min(),
                        end=df_cfips_group['first_day_of_month'].max(),
                        predictions=[
                            f'{fold}_last_value_predictions',
                            f'{fold}_last_value_adjusted_predictions',
                            f'{fold}_last_value_adjusted_weighted_predictions'
                        ],
                        score=True,
                        path=val_predictions_visualizations_directory / f'{cfips}_{fold}.png'
                    )

        # Display validation scores
        df_last_value_scores = pd.DataFrame(last_value_scores)
        df_last_value_adjusted_scores = pd.DataFrame(last_value_adjusted_scores)
        df_last_value_adjusted_weighted_scores = pd.DataFrame(last_value_adjusted_weighted_scores)

        for df, name in zip([df_last_value_scores, df_last_value_adjusted_scores, df_last_value_adjusted_weighted_scores],
                            ['last_value_scores', 'last_value_adjusted_scores', 'last_value_adjusted_weighted_scores']):

            for fold, scores in df.iterrows():
                settings.logger.info(f'Fold {int(fold) + 1} - {name} Scores: {json.dumps(scores.to_dict(), indent=2)}')
            settings.logger.info(
                f'''
                Mean Validation Scores
                {json.dumps(df.mean(axis=0).to_dict(), indent=2)}
                and Standard Deviations
                ±{json.dumps(df.std(axis=0).to_dict(), indent=2)}
                '''
            )

            if config['persistence']['visualize_scores']:
                visualization.visualize_scores(
                    df_scores=df,
                    path=model_directory / f'{name}.png'
                )
                settings.logger.info(f'Saved {name}.png to {model_directory}')

    elif args.mode == 'submission':

        submission_directory = Path(settings.DATA / 'submissions')
        submission_directory.mkdir(parents=True, exist_ok=True)

        df = pd.concat((
            df_train,
            df_test.groupby('cfips').tail(6)
        ), axis=0, ignore_index=True)
        df.sort_values(by=['cfips', 'first_day_of_month'], ascending=[True, True], inplace=True)
        df = df.reset_index(drop=True)

        last_values = df.groupby('cfips')['microbusiness_density'].last()
        df.loc[df['first_day_of_month'] >= '2022-11-01', 'last_value_predictions'] = df.loc[df['first_day_of_month'] >= '2022-11-01', 'cfips'].map(last_values)

        df_acs_5y = pd.read_parquet(settings.DATA / 'external' / 'processed' / 'acs_5y.parquet')
        df_acs_5y['year'] = df_acs_5y['first_day_of_year'].dt.year + 2
        df['year'] = df['first_day_of_month'].dt.year
        df = df.merge(df_acs_5y, on=['cfips', 'year'], how='left')
        adjustment_ratios = df_acs_5y.loc[df_acs_5y['first_day_of_year'] == '2020-01-01'].set_index('cfips')['S0101_C01_026E'] /\
                            df_acs_5y.loc[df_acs_5y['first_day_of_year'] == '2021-01-01'].set_index('cfips')['S0101_C01_026E']
        df['adjustment_ratio'] = df['cfips'].map(adjustment_ratios)
        df['last_value_adjusted_predictions'] = df['last_value_predictions'] * df['adjustment_ratio']

        for cfips, df_cfips_group in tqdm(df.groupby('cfips'), total=df['cfips'].nunique()):

            n = 5
            last_values = df_cfips_group.loc[df_cfips_group[target].notna(), target].values[-n:]
            last_values_diff = np.diff(last_values)

            if np.all(last_values_diff > 0):
                weights = np.array([1.0, 1.0, 1.0, 1.0, 1.010, 1.015, 1.015, 1.0])
            elif np.all(last_values_diff < 0):
                weights = np.array([1.0, 1.0, 1.0, 1.0, 0.998, 0.994, 0.990, 1.0])
            else:
                weights = np.array([1.0, 1.0, 1.0, 1.0, 1.010, 1.010, 1.010, 1.0])

            df.loc[df_cfips_group.loc[df_cfips_group['last_value_adjusted_predictions'].notna()].index, 'last_value_adjusted_weighted_predictions'] = df_cfips_group.loc[df_cfips_group['last_value_adjusted_predictions'].notna(), 'last_value_adjusted_predictions'] * weights

        settings.logger.info('Finished predicting test set')

        if config['persistence']['visualize_test_predictions']:

            test_predictions_visualizations_directory = model_directory / 'test_predictions'
            test_predictions_visualizations_directory.mkdir(parents=True, exist_ok=True)

            for cfips, df_cfips_group in tqdm(df.groupby('cfips'), total=df['cfips'].nunique()):
                visualization.visualize_timeseries_target(
                    df=df_cfips_group,
                    datetime='first_day_of_month',
                    target='microbusiness_density',
                    cfips=cfips,
                    start=df_cfips_group['first_day_of_month'].min(),
                    end=df_cfips_group['first_day_of_month'].max(),
                    predictions=[
                        'last_value_predictions',
                        'last_value_adjusted_predictions',
                        'last_value_adjusted_weighted_predictions'
                    ],
                    score=False,
                    path=test_predictions_visualizations_directory / f'{cfips}.png'
                )

            settings.logger.info('Finished visualizing test set predictions')

        df_test = df.groupby('cfips').tail(8).reset_index(drop=True)
        df_test['microbusiness_density'] = df_test['last_value_adjusted_weighted_predictions']
        df_test[['row_id', 'microbusiness_density']].to_csv(submission_directory / 'last_value_adjusted_weighted_submission.csv', index=False)
        settings.logger.info(f'Saved submission to {submission_directory}')

    else:
        raise ValueError(f'Invalid mode {args.mode}')
