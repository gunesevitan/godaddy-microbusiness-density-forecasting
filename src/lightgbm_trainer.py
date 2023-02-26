import argparse
from pathlib import Path
import yaml
import json
from tqdm import tqdm
import numpy as np
import pandas as pd
import lightgbm as lgb

import settings
import metrics
import feature_engineering
import visualization



def smape_eval_lgb(y_pred, train_dataset):

    """
    Calculate SMAPE metric score while training a LightGBM model

    Parameters
    ----------
    y_pred: array-like of shape (n_samples)
        Predictions arrays

    train_dataset: lightgbm.Dataset
        Training dataset

    Returns
    -------
    eval_name: str
        Name of the evaluation metric

    eval_result: float
        Result of the evaluation metric

    is_higher_better: bool
        Whether higher is better or worse for the evaluation metric
    """

    eval_name = 'smape'
    y_true = train_dataset.get_label()
    eval_result = metrics.symmetric_mean_absolute_percentage_error(y_true, y_pred)
    is_higher_better = False

    return eval_name, eval_result, is_higher_better


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

    df_train = feature_engineering.create_features(
        df=df_train
    )

    features = config['training']['features']
    target = config['training']['target']
    folds = config['training']['folds']

    if args.mode == 'validation':

        settings.logger.info('Running LightGBM model for training')

        df_feature_importance_gain = pd.DataFrame(
            data=np.zeros((len(features), len(folds))),
            index=features,
            columns=folds
        )
        df_feature_importance_split = pd.DataFrame(
            data=np.zeros((len(features), len(folds))),
            index=features,
            columns=folds
        )
        scores = []

        for fold in folds:

            train_idx, val_idx = df_train.loc[df_train[fold] == 0].index, df_train.loc[df_train[fold] == 1].index
            if len(val_idx) == 0:
                val_idx = train_idx

            settings.logger.info(
                f'''
                {fold}
                Training: ({len(train_idx)}, {len(features)}) - [{df_train.loc[train_idx, 'first_day_of_month'].min()} - {df_train.loc[train_idx, 'first_day_of_month'].max()})
                Validation: ({len(val_idx)}, {len(features)}) - [{df_train.loc[val_idx, 'first_day_of_month'].min()} - {df_train.loc[val_idx, 'first_day_of_month'].max()})
                '''
            )

            train_dataset = lgb.Dataset(df_train.loc[train_idx, features], label=df_train.loc[train_idx, target], categorical_feature=config['training']['categorical_features'])
            val_dataset = lgb.Dataset(df_train.loc[val_idx, features], label=df_train.loc[val_idx, target], categorical_feature=config['training']['categorical_features'])

            # Set model parameters, train parameters, callbacks and start training
            model = lgb.train(
                params=config['model'],
                train_set=train_dataset,
                valid_sets=[train_dataset, val_dataset],
                num_boost_round=config['fit']['boosting_rounds'],
                callbacks=[
                    lgb.early_stopping(config['fit']['early_stopping']),
                    lgb.log_evaluation(config['fit']['log_evaluation'])
                ],
                feval=[smape_eval_lgb]
            )
            # Save trained model
            if config['persistence']['save_models']:
                model.save_model(
                    model_directory / f'model_{fold}.lgb',
                    num_iteration=None,
                    start_iteration=0,
                    importance_type='gain'
                )
                settings.logger.info(f'Saved model_{fold}.lgb to {model_directory}')

            df_feature_importance_gain[fold] = model.feature_importance(importance_type='gain')
            df_feature_importance_split[fold] = model.feature_importance(importance_type='split')

            val_predictions = model.predict(df_train.loc[val_idx, features])
            df_train.loc[val_idx, f'{fold}_predictions'] = val_predictions
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

        settings.logger.info('Running LightGBM model for evaluation')

        # Display validation scores
        df_scores = pd.DataFrame(scores)
        for fold, scores in df_scores.iterrows():
            settings.logger.info(f'Fold {int(fold) + 1} - Validation Scores: {json.dumps(scores.to_dict(), indent=2)}')
        settings.logger.info(
            f'''
            LightGBM Mean Validation Scores
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

        for importance_type, df_feature_importance in zip(['gain', 'split'], [df_feature_importance_gain, df_feature_importance_split]):
            df_feature_importance['mean'] = df_feature_importance[folds].mean(axis=1)
            df_feature_importance['std'] = df_feature_importance[folds].std(axis=1).fillna(0)
            df_feature_importance.sort_values(by='mean', ascending=False, inplace=True)
            if config['persistence']['visualize_feature_importance'] is not None:
                visualization.visualize_feature_importance(
                    df_feature_importance=df_feature_importance,
                    path=model_directory / f'feature_importance_{importance_type}.png'
                )
                settings.logger.info(f'Saved feature_importance_{importance_type}.png to {model_directory}')

    elif args.mode == 'submission':

        pass

    else:
        raise ValueError(f'Invalid mode {args.mode}')
