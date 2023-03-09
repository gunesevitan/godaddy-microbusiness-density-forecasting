from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import settings
import metrics


def visualize_timeseries_target(df, datetime, target, name, start, end, predictions=None, score=False, path=None):

    """
    Visualize time series target on specified period

    Parameters
    ----------
    df: pandas.DataFrame
        Dataframe with given datetime and target columns

    datetime: str
        Name of the datetime column

    target: str
        Name of the target column

    name: str
        Group name for the title

    start: str
        Start date

    end: str
        End date

    predictions: str
        Name of the predictions column

    score: bool
        Whether to calculate predictions scores or not

    path: path-like str or None
        Path of the output file or None (if path is None, plot is displayed with selected backend)
    """

    date_idx = (df[datetime] >= start) & (df[datetime] < end)

    fig, ax = plt.subplots(figsize=(24, 6), dpi=100)
    ax.plot(df.loc[date_idx].set_index(datetime)[target], '-o', linewidth=2)
    ax_population = ax.twinx()
    ax_population.plot(df.loc[date_idx].set_index(datetime)['S0101_C01_026E'], linewidth=1, alpha=0.5)
    ax.axvline(pd.to_datetime('2022-01-01 00:00:00'), color='r')
    ax.axvline(pd.to_datetime('2021-01-01 00:00:00'), color='r')
    ax.axvline(pd.to_datetime('2020-01-01 00:00:00'), color='r')

    if predictions is not None:
        for prediction in predictions:
            if score:
                val_idx = df[prediction].notna()
                prediction_scores = metrics.regression_scores(y_true=df.loc[(date_idx & val_idx), target], y_pred=df.loc[(date_idx & val_idx), prediction])
                label = f'{prediction} - SMAPE: {prediction_scores["mean_absolute_error"]:.4f}'
            else:
                label = prediction

            ax.plot(
                df.loc[date_idx].set_index(datetime)[prediction].dropna(),
                '-o',
                linewidth=2,
                label=label
            )
            ax.legend(prop={'size': 18})

    ax.tick_params(axis='x', labelsize=12.5, pad=10)
    ax.tick_params(axis='y', labelsize=12.5, pad=10)
    ax.set_title(f'{name} [{start}, {end}) - {target}', size=20, pad=15)

    if path is None:
        plt.show()
    else:
        plt.savefig(path)
        plt.close(fig)


def visualize_categorical_feature_distribution(df, feature, path=None):

    """
    Visualize distribution of given categorical column in given dataframe

    Parameters
    ----------
    df: pandas.DataFrame
        Dataframe with given feature column

    feature: str
        Name of the categorical feature

    path: path-like str or None
        Path of the output file or None (if path is None, plot is displayed with selected backend)
    """

    fig, ax = plt.subplots(figsize=(24, df[feature].value_counts().shape[0] + 4), dpi=100)
    sns.barplot(
        y=df[feature].value_counts().values,
        x=df[feature].value_counts().index,
        color='tab:blue',
        ax=ax
    )
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticklabels([
        f'{x} ({value_count:,})' for value_count, x in zip(
            df[feature].value_counts().values,
            df[feature].value_counts().index
        )
    ])
    ax.tick_params(axis='x', labelsize=15, pad=10)
    ax.tick_params(axis='y', labelsize=15, pad=10)
    ax.set_title(f'Value Counts {feature}', size=20, pad=15)

    if path is None:
        plt.show()
    else:
        plt.savefig(path)
        plt.close(fig)


def visualize_continuous_feature_distribution(df_train, df_test, feature, path=None):

    """
    Visualize distribution of given continuous column in given dataframe(s)

    Parameters
    ----------
    df_train: pandas.DataFrame
        Training dataframe with given feature column

    df_test: pandas.DataFrame or None
        Test dataframe with given feature column

    feature: str
        Name of the continuous feature

    path: path-like str or None
        Path of the output file or None (if path is None, plot is displayed with selected backend)
    """

    fig, ax = plt.subplots(figsize=(24, 6), dpi=100)
    if df_test is not None:
        sns.kdeplot(df_train[feature], label='train', fill=True, log_scale=False, ax=ax)
        sns.kdeplot(df_test[feature], label='test', fill=True, log_scale=False, ax=ax)
    else:
        sns.kdeplot(df_train[feature], fill=True, log_scale=False, ax=ax)

    ax.tick_params(axis='x', labelsize=12.5)
    ax.tick_params(axis='y', labelsize=12.5)
    ax.set_xlabel('')
    ax.set_ylabel('')
    if df_test is not None:
        ax.legend(prop={'size': 15})
        title = f'''
        {feature}
        Mean - Train: {df_train[feature].mean():.2f} |  Test: {df_test[feature].mean():.2f}
        Median - Train: {df_train[feature].median():.2f} |  Test: {df_test[feature].median():.2f}
        Std - Train: {df_train[feature].std():.2f} |  Test: {df_test[feature].std():.2f}
        Min - Train: {df_train[feature].min():.2f} |  Test: {df_test[feature].min():.2f}
        Max - Train: {df_train[feature].max():.2f} |  Test: {df_test[feature].max():.2f}
        '''
    else:
        title = f'''
        {feature}
        Mean - Train: {df_train[feature].mean():.2f}
        Median - Train: {df_train[feature].median():.2f}
        Std - Train: {df_train[feature].std():.2f}
        Min - Train: {df_train[feature].min():.2f}
        Max - Train: {df_train[feature].max():.2f}
        '''
    ax.set_title(title, size=20, pad=12.5)

    if path is None:
        plt.show()
    else:
        plt.savefig(path, bbox_inches='tight')
        plt.close(fig)


def visualize_feature_importance(df_feature_importance, path=None):

    """
    Visualize feature importance in descending order

    Parameters
    ----------
    df_feature_importance: pandas.DataFrame of shape (n_features, n_splits)
        Dataframe of feature importance

    path: path-like str or None
        Path of the output file or None (if path is None, plot is displayed with selected backend)
    """

    n_splits = df_feature_importance.shape[1] - 2

    fig, ax = plt.subplots(figsize=(24, 20), dpi=100)
    ax.barh(
        range(len(df_feature_importance)),
        df_feature_importance['mean'],
        xerr=df_feature_importance['std'],
        ecolor='black',
        capsize=10,
        align='center',
    )
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_yticks(range(len(df_feature_importance)))
    ax.set_yticklabels([f'{k} ({v:.2f})' for k, v in df_feature_importance['mean'].to_dict().items()])
    ax.tick_params(axis='x', labelsize=15, pad=10)
    ax.tick_params(axis='y', labelsize=15, pad=10)
    ax.set_title(f'Mean and Std Feature Importance of {n_splits} Models', size=20, pad=15)
    plt.gca().invert_yaxis()

    if path is None:
        plt.show()
    else:
        plt.savefig(path, bbox_inches='tight')
        plt.close(fig)


def visualize_scores(df_scores, path=None):

    """
    Visualize feature coefficients in descending order

    Parameters
    ----------
    df_scores: pandas.DataFrame of shape (n_splits, n_metrics)
        Dataframe with multiple scores and metrics

    path: path-like str or None
        Path of the output file or None (if path is None, plot is displayed with selected backend)
    """

    # Create mean and std of scores for error bars
    df_scores = df_scores.T
    n_scores = df_scores.shape[1]
    column_names = df_scores.columns.to_list()
    df_scores['mean'] = df_scores[column_names].mean(axis=1)
    df_scores['std'] = df_scores[column_names].std(axis=1).fillna(0)

    fig, ax = plt.subplots(figsize=(32, 8))
    ax.barh(
        y=np.arange(df_scores.shape[0]),
        width=df_scores['mean'],
        xerr=df_scores['std'],
        align='center',
        ecolor='black',
        capsize=10
    )
    ax.set_yticks(np.arange(df_scores.shape[0]))
    ax.set_yticklabels([
        f'{metric}\n{mean:.4f} (±{std:.4f})' for metric, mean, std in zip(
            df_scores.index,
            df_scores['mean'].values,
            df_scores['std'].values
        )
    ])
    ax.set_xlabel('')
    ax.tick_params(axis='x', labelsize=15, pad=10)
    ax.tick_params(axis='y', labelsize=15, pad=10)
    ax.set_title(f'Mean and Std Scores of {n_scores} Model(s)', size=20, pad=15)

    if path is None:
        plt.show()
    else:
        plt.savefig(path)
        plt.close(fig)


if __name__ == '__main__':

    df_train = pd.read_parquet(settings.DATA / 'train.parquet')
    df_train = df_train.merge(pd.read_parquet(settings.DATA / 'census_starter.parquet'))
    settings.logger.info(f'Training Set Shape: {df_train.shape} - Memory Usage: {df_train.memory_usage().sum() / 1024 ** 2:.2f}')

    df_test = pd.read_parquet(settings.DATA / 'test.parquet')
    df_test = df_test.merge(pd.read_parquet(settings.DATA / 'census_starter.parquet'))
    settings.logger.info(f'Test Set Shape: {df_test.shape} - Memory Usage: {df_test.memory_usage().sum() / 1024 ** 2:.2f}')

    VISUALIZE_TIME_SERIES_TARGETS = True
    VISUALIZE_CONTINUOUS_FEATURE_DISTRIBUTIONS = True

    if VISUALIZE_TIME_SERIES_TARGETS:

        time_series_target_visualization_directory = settings.EDA / 'time_series_target'
        time_series_target_visualization_directory.mkdir(parents=True, exist_ok=True)

        for cfips, df_cfips_group in tqdm(df_train.groupby('cfips'), total=df_train['cfips'].nunique()):
            df_cfips_group = df_cfips_group.sort_values(by='first_day_of_month', ascending=True)
            visualize_timeseries_target(
                df=df_cfips_group,
                datetime='first_day_of_month',
                target='microbusiness_density',
                name=cfips,
                start=df_cfips_group['first_day_of_month'].min(),
                end=df_cfips_group['first_day_of_month'].max(),
                path=time_series_target_visualization_directory / f'{cfips}.png'
            )

    if VISUALIZE_CONTINUOUS_FEATURE_DISTRIBUTIONS:

        continuous_feature_distribution_visualization_directory = settings.EDA / 'continuous_feature_distribution'
        continuous_feature_distribution_visualization_directory.mkdir(parents=True, exist_ok=True)

        continuous_features = [
            'microbusiness_density', 'active', 'month', 'year', 'pct_bb_2017',
            'pct_bb_2018', 'pct_bb_2019', 'pct_bb_2020', 'pct_bb_2021',
            'pct_college_2017', 'pct_college_2018', 'pct_college_2019',
            'pct_college_2020', 'pct_college_2021', 'pct_foreign_born_2017',
            'pct_foreign_born_2018', 'pct_foreign_born_2019',
            'pct_foreign_born_2020', 'pct_foreign_born_2021', 'pct_it_workers_2017',
            'pct_it_workers_2018', 'pct_it_workers_2019', 'pct_it_workers_2020',
            'pct_it_workers_2021', 'median_hh_inc_2017', 'median_hh_inc_2018',
            'median_hh_inc_2019', 'median_hh_inc_2020', 'median_hh_inc_2021'
        ]

        for continuous_feature in tqdm(continuous_features):
            visualize_continuous_feature_distribution(
                df_train=df_train,
                df_test=None,
                feature=continuous_feature,
                path=continuous_feature_distribution_visualization_directory / f'{continuous_feature}.png'
            )
