import sys
import argparse
from pathlib import Path
from tqdm import tqdm
import yaml
import json
import pandas as pd
import matplotlib.pyplot as plt
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
from skopt.plots import plot_convergence

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

    if args.mode == 'validation':

        settings.logger.info('Running last value model in validation mode')

        target = config['training']['target']
        folds = config['training']['folds']

        search_space = [
            Real(config['optimization']['lower_bound'], config['optimization']['upper_bound'], name=str(cfips)) for cfips in df_train['cfips'].unique()
        ]

        @use_named_args(search_space)
        def validation(**kwargs):

            scores = []

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

                last_values = df_train.loc[train_idx].groupby('cfips')[target].last().to_dict()
                for cfips, last_value in last_values.items():
                    last_values[cfips] *= kwargs[str(cfips)]

                df_train.loc[val_idx, f'{fold}_predictions'] = df_train.loc[val_idx, 'cfips'].map(last_values)
                val_scores = metrics.regression_scores(
                    y_true=df_train.loc[val_idx, target],
                    y_pred=df_train.loc[val_idx, f'{fold}_predictions'],
                )
                scores.append(val_scores)
                settings.logger.info(f'{fold} - Validation Scores: {json.dumps(val_scores, indent=2)}')

            # Display validation scores
            df_scores = pd.DataFrame(scores)
            for fold, scores in df_scores.iterrows():
                settings.logger.info(f'Fold {int(fold) + 1} - Validation Scores: {json.dumps(scores.to_dict(), indent=2)}')
            settings.logger.info(
                f'''
                Mean Validation Scores
                {json.dumps(df_scores.mean(axis=0).to_dict(), indent=2)}
                and Standard Deviations
                ±{json.dumps(df_scores.std(axis=0).to_dict(), indent=2)}
                '''
            )
            smape = df_scores.mean(axis=0).to_dict()['smape']

            return smape

        result = gp_minimize(
            func=validation,
            dimensions=search_space,
            n_calls=config['optimization']['n_calls'],
            n_initial_points=config['optimization']['n_initial_points'],
            acq_func=config['optimization']['acq_func'],
            acq_optimizer=config['optimization']['acq_optimizer'],
            random_state=config['optimization']['random_state'],
            n_jobs=config['optimization']['n_jobs'],
        )
        settings.logger.info(
            f'''
            Finished optimizing function validation
            Minimum value {result['fun']:.6f}
            '''
        )

        if config['persistence']['visualize_convergence']:
            fig, ax = plt.subplots(figsize=(16, 8))
            plot_convergence(result)
            plt.savefig(model_directory / 'convergence.png')
            plt.close(fig)

        df_multipliers = df_train[['cfips']].drop_duplicates().reset_index(drop=True)
        df_multipliers['multiplier'] = result['x']
        df_multipliers.to_csv(model_directory / 'multipliers.csv', index=False)
        settings.logger.info(f'Saved multipliers.csv to {model_directory}')

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
        multipliers = pd.read_csv(model_directory / 'multipliers.csv')
        last_values = last_values * multipliers['multiplier'].values
        df.loc[df['first_day_of_month'] >= '2023-01-01', 'predictions'] = df.loc[df['first_day_of_month'] >= '2023-01-01', 'cfips'].map(last_values)

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
                    predictions='predictions',
                    score=False,
                    path=test_predictions_visualizations_directory / f'{cfips}.png'
                )

            settings.logger.info('Finished visualizing test set predictions')

        df_test = df.groupby('cfips').tail(6).reset_index(drop=True)
        df_test['microbusiness_density'] = df_test['predictions']
        df_test[['row_id', 'microbusiness_density']].to_csv(submission_directory / 'bayesian_optimization_submission.csv', index=False)
        settings.logger.info(f'Saved bayesian_optimization_submission.csv to {submission_directory}')

    else:
        raise ValueError(f'Invalid mode {args.mode}')
