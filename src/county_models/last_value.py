import sys
import argparse
from pathlib import Path
import yaml
import json
from tqdm import tqdm
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

            df_train.loc[val_idx, f'{fold}_predictions'] = df_train.loc[val_idx, 'cfips'].map(df_train.loc[train_idx].groupby('cfips')[target].nth(-1))
            val_scores = metrics.regression_scores(
                y_true=df_train.loc[val_idx, target],
                y_pred=df_train.loc[val_idx, f'{fold}_predictions'],
            )
            scores.append(val_scores)
            settings.logger.info(f'{fold} - Validation Scores: {json.dumps(val_scores, indent=2)}')

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
                        predictions=f'{fold}_predictions',
                        path=val_predictions_visualizations_directory / f'{cfips}_{fold}.png'
                    )

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

        if config['persistence']['visualize_scores']:
            visualization.visualize_scores(
                df_scores=df_scores,
                path=model_directory / 'scores.png'
            )
            settings.logger.info(f'Saved scores.png to {model_directory}')

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
        df_test[['row_id', 'microbusiness_density']].to_csv(submission_directory / 'last_value_submission.csv', index=False)
        settings.logger.info(f'Saved last_value_submission.csv to {submission_directory}')

    else:
        raise ValueError(f'Invalid mode {args.mode}')